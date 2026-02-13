from collections import defaultdict
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import asyncio
import time

from config.logger import logger

from schemas.classes import Service
from src.pipeline import RAGPipeline
from src.cache import LRUPipelineCache
from db.functions import (
    get_all_services_from_db, 
    get_service_by_name_from_db, 
    create_services_in_db, 
    delete_service_from_db
)
from config.settings import VECTORSTORE_PATH, DOCS_FOLDER_PATH


class MultiServiceManager:
    """Manages multiple RAG pipelines for different services with LRU caching"""
    
    def __init__(self, max_concurrent_builds: int = 10, cache_size: int = 10, retriever_type: str = "SemanticRetriever"):  
        self.customer_tokens: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        self.pipeline_cache = LRUPipelineCache(max_size=cache_size)
        self._lock = asyncio.Lock()
        self._initialized = False
        self.max_concurrent_builds = max_concurrent_builds
        self.retriever_type = retriever_type
        
        logger.info(f"🛠️ MultiServiceManager instantiated (max_concurrent={max_concurrent_builds}, cache={cache_size})")

    # ---------------- INITIALIZATION ---------------- #

    async def initialize_async(self) -> bool:
        """Creates all pipelines on disk during startup"""
        async with self._lock:
            if self._initialized:
                logger.warning("⚠️ Manager already initialized")
                return True
                
            success = await self._create_all_pipelines_on_disk()
            
            if not success:
                logger.error("❌ Error during MultiServiceManager initialization")
                return False
            
            self._initialized = True
            logger.info("✅ MultiServiceManager initialized (pipelines on disk, cache empty)")
            return True

    # ---------------- ON-DISK PIPELINE MANAGEMENT ---------------- #

    async def _create_all_pipelines_on_disk(self) -> bool:
        """Internal method to generate all pipelines on disk"""
        logger.info("⚙️ Creating pipelines on disk...")
        start = time.time()

        try:
            services_data = get_all_services_from_db()
            if not services_data:
                logger.info("ℹ️ No services found in database")
                return True
            logger.info(f"🔍 Found {len(services_data)} services in database")

            # Create Service objects and load tokens into memory
            services = []
            for service_dict in services_data:
                try:
                    service = Service.from_dict(service_dict)
                    services.append(service)
                    
                    # Update customer_tokens - use set logic to avoid duplicates
                    if service.name not in self.customer_tokens[service.customer_name]:
                        self.customer_tokens[service.customer_name][service.name] = []
                    
                    self.customer_tokens[service.customer_name][service.name].extend(
                        [token for token in service.tokens if token not in self.customer_tokens[service.customer_name][service.name]]
                    )
                except Exception as e:
                    logger.error(f"❌ Error parsing service {service_dict.get('name', 'unknown')}: {e}")
                    continue

            if not services:
                logger.error("❌ No valid services found in database")
                return False

            vectorstore_path = Path(VECTORSTORE_PATH)
            if vectorstore_path.exists() and vectorstore_path.is_dir():
                logger.info(f"🗑️ Delete old vectorstore: {vectorstore_path}")
                await asyncio.to_thread(shutil.rmtree, vectorstore_path)
            
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 New Folder vectorstore ready: {vectorstore_path}")
            
            logger.info(f"🔨 Building {len(services)} pipelines...")
            semaphore = asyncio.Semaphore(min(self.max_concurrent_builds, len(services)))
            
            async def build_with_limit(service_obj: Service):
                async with semaphore:
                    return await self._create_service_pipeline(service_obj)

            tasks = [build_with_limit(service) for service in services]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            created = 0
            failed = 0
            for i, result in enumerate(results):
                service_name = services[i].name if i < len(services) else f"service_{i}"
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"❌ Error creating pipeline for '{service_name}': {result}")
                elif result:
                    created += 1
                else:
                    failed += 1
                    logger.error(f"❌ Creation failed for '{service_name}' without exception")

            logger.info(f"✅ Pipelines created: {created}, failed: {failed}")

            elapsed = time.time() - start
            logger.info(f"🔄 Initialization completed in {elapsed:.2f}s")
            return created > 0 or len(services) == 0 
        
        except Exception as e:
            logger.error(f"❌ Error creating pipelines on disk: {e}", exc_info=True)
            return False

    async def _create_service_pipeline(self, service: Service) -> bool:
        """Creates a single pipeline on disk for a specific service"""
        logger.info(f"🔨 Building pipeline '{service.name}' for '{service.customer_name}'")
        start_time = time.time()

        try:
            if not service.documents or len(service.documents) == 0:
                logger.error(f"❌ No documents provided for '{service.name}'")
                return False

            # Create temporary pipeline
            pipeline = RAGPipeline(service.name, service.customer_name, retriever_type=self.retriever_type)
            
            # Load documents
            upload_docs = await pipeline.load_documents_batch(service.documents)
            if len(upload_docs) == 0:
                logger.error(f"❌ No documents loaded for {service.name}")
                return False
            
            # Build index
            success = await pipeline.build_index_async()
            if not success:
                logger.error(f"❌ Error building index for {service.name}")
                return False

            # Save to disk
            service_path = self._get_service_vectorstore_path(service.name, service.customer_name)
            service_path.mkdir(parents=True, exist_ok=True)

            success = await pipeline.save_pipeline(str(service_path))
            if not success:
                logger.error(f"❌ Error saving pipeline for {service.name}")
                return False

            # IMPORTANT: Delete pipeline to free memory
            del pipeline
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Pipeline '{service.name}' created on disk in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building pipeline {service.name}: {e}", exc_info=True)
            return False

    # ---------------- DISK LOADING WITH CACHE ---------------- #

    def _get_service_key(self, customer_name: str, service_name: str) -> str:
        """Generates unique cache key"""
        return f"{customer_name}::{service_name}"

    async def _get_or_load_pipeline(self, customer_name: str, service_name: str) -> Optional[RAGPipeline]:
        """Retrieves pipeline from cache or loads it from disk if missing"""
        service_key = self._get_service_key(customer_name, service_name)
        
        # Check cache
        pipeline = await self.pipeline_cache.get(service_key)
        if pipeline:
            return pipeline
        
        # Load from disk
        logger.info(f"💿 Loading pipeline '{service_name}' for '{customer_name}' from disk...")
        start = time.time()
        
        try:
            service_path = self._get_service_vectorstore_path(service_name, customer_name)
            
            if not service_path.exists():
                logger.error(f"❌ Pipeline '{service_name}' not found on disk")
                return None
            
            # Instantiate and load
            pipeline = RAGPipeline(service_name, customer_name, retriever_type=self.retriever_type)
            success = await pipeline.load_pipeline(str(service_path))
            
            if not success:
                logger.error(f"❌ Failed to load pipeline '{service_name}'")
                return None
            
            # Add to cache (automatically handles LRU eviction)
            await self.pipeline_cache.put(service_key, pipeline)
            
            elapsed = time.time() - start
            logger.info(f"✅ Pipeline '{service_name}' loaded in {elapsed:.2f}s")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Error loading pipeline '{service_name}': {e}", exc_info=True)
            return None

    # ---------------- CORE FUNCTIONS ---------------- #

    def _get_service_vectorstore_path(self, service_name: str, customer_name: str) -> Path:
        """Returns the path to the vectorstore folder associated with the service"""
        return Path(VECTORSTORE_PATH) / customer_name.strip() / service_name.strip()

    async def save_service(self, service: Service) -> bool:
        """
        Creates a new service or updates it with specified documents.
        The pipeline is saved to disk but NOT loaded into memory.
        """
        logger.info(f"🆕 Saving service: '{service.name}' for {service.customer_name}")
        start_time = time.time()
        
        try:
            result = get_service_by_name_from_db(service_name=service.name, customer_name=service.customer_name)
            
            if result is not None:
                logger.info(f"⚠️ Service '{service.name}' already exists, evaluating changes...")
                existing_service = Service.from_dict(result)

                # Metadata only changes (no rebuild required)
                if service.documents == existing_service.documents:
                    logger.info(f"⚠️ No changes in documents for '{service.name}'")
                    
                    if service.description != existing_service.description or service.tokens != existing_service.tokens:
                        logger.info(f"✏️ Updating metadata for '{service.name}'")
                        success = create_services_in_db(service)
                        if success:
                            # Update tokens in memory
                            if service.customer_name not in self.customer_tokens:
                                self.customer_tokens[service.customer_name] = {}
                            self.customer_tokens[service.customer_name][service.name] = service.tokens
                        return success
                    return True
                
                # Documents changed: rebuild required
                logger.info(f"⚙️ Rebuilding pipeline '{service.name}'")
                
                # Remove from cache if present
                service_key = self._get_service_key(service.customer_name, service.name)
                await self.pipeline_cache.remove(service_key)
                
                # Delete old pipeline from disk
                service_path = self._get_service_vectorstore_path(service.name, service.customer_name)
                if service_path.exists():
                    await asyncio.to_thread(shutil.rmtree, service_path)

            # Build new pipeline on disk
            success = await self._create_service_pipeline(service)
            if not success:
                logger.error(f"❌ Could not create pipeline for '{service.name}'")
                return False

            # Save to database
            logger.info(f"💾 Saving service '{service.name}' to database...")
            success = create_services_in_db(service)

            if not success:
                logger.error(f"❌ Error saving service '{service.name}' to database")
                # Cleanup on failure
                service_path = self._get_service_vectorstore_path(service.name, service.customer_name)
                if service_path.exists():
                    await asyncio.to_thread(shutil.rmtree, service_path)
                return False
            
            # Update tokens in memory
            if service.customer_name not in self.customer_tokens:
                self.customer_tokens[service.customer_name] = {}
            self.customer_tokens[service.customer_name][service.name] = service.tokens
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Service '{service.name}' saved in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving service '{service.name}': {e}", exc_info=True)
            return False

    async def search_query(self, customer_name: str, service_name: str, query: str) -> Optional[Dict[str, Dict]]:
        """
        Executes a search on a specific service.
        Automatically loads pipeline from disk if not cached.
        """
        start_time = time.time()
        try:
            # Get pipeline (cache or disk)
            pipeline = await self._get_or_load_pipeline(customer_name, service_name)
            if not pipeline:
                logger.error(f"❌ Failed to load pipeline '{service_name}'")
                return []

            logger.info(f"🔍 Searching service '{service_name}': '{query}'")

            result = await pipeline.search(query)
            elapsed = time.time() - start_time
            logger.info(f"✅ Found {len(result)} results in {elapsed:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Search error for query '{query}' on service '{service_name}': {e}", exc_info=True)
            return []

    async def delete_service(self, service_name: str, customer_name: str) -> bool:
        """Deletes a specific service from disk, cache, and database"""
        logger.info(f"🗑️ Deleting service '{service_name}' for '{customer_name}'")
        start_time = time.time()
        
        try:
            result = get_service_by_name_from_db(service_name, customer_name)
            if result is None:
                logger.warning(f"⚠️ Service '{service_name}' not found in database")
                return False

            async with self._lock:
                # Remove from cache
                service_key = self._get_service_key(customer_name, service_name)
                await self.pipeline_cache.remove(service_key)
                
                # Delete from disk
                service_path = self._get_service_vectorstore_path(service_name, customer_name)
                if service_path.exists():
                    await asyncio.to_thread(shutil.rmtree, service_path)
                    logger.debug(f"🗑️ Pipeline deleted from disk")
                
                # Delete from database
                success = delete_service_from_db(service_name, customer_name)
                if not success:
                    logger.error(f"❌ Error deleting from database")
                    return False
                
                # Clear metadata from memory
                if customer_name in self.customer_tokens and service_name in self.customer_tokens[customer_name]:
                    del self.customer_tokens[customer_name][service_name]
                
                # Delete log files
                log_path = Path("logs") / customer_name / f"{service_name}.log"
                if log_path.exists():
                    log_path.unlink()
                    logger.debug(f"🗑️ Log file deleted")

                elapsed = time.time() - start_time
                logger.info(f"✅ Service '{service_name}' deleted in {elapsed:.2f}s")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error deleting service {service_name}: {e}", exc_info=True)
            return False

    # ---------------- STATS AND MANAGEMENT ---------------- #
    
    async def get_cache_stats(self) -> Dict:
        """Returns statistics on the pipeline cache"""
        return self.pipeline_cache.get_cache_stats()

    async def preload_most_used_services(self, service_list: List[Tuple[str, str]]):
        """Pre-loads the most frequently used services into cache at startup"""
        logger.info(f"🔄 Pre-loading {len(service_list)} most used services...")
        
        for customer_name, service_name in service_list[:self.pipeline_cache.max_size]:
            await self._get_or_load_pipeline(customer_name, service_name)
        
        logger.info("✅ Pre-loading completed")

    async def get_available_documents(self) -> List[str]:
        """Returns a list of available documents in the docs folder"""
        docs_path = Path(DOCS_FOLDER_PATH)
        if not docs_path.exists():
            logger.warning(f"⚠️ Documents folder not found: {docs_path}")
            return []
        
        try:
            documents = [
                f.name for f in docs_path.iterdir() 
                if f.is_file() and not f.name.startswith('.')
            ]
            return sorted(documents)
        except Exception as e:
            logger.error(f"❌ Error reading documents: {e}", exc_info=True)
            return []

    async def warmup_cache(self, customer_name: str, service_names: List[str]):
        """Warams up the cache for specific services"""
        logger.info(f"🔥 Cache warmup for {len(service_names)} services under '{customer_name}'")
        
        for service_name in service_names:
            await self._get_or_load_pipeline(customer_name, service_name)
        
        logger.info("✅ Warmup completed")
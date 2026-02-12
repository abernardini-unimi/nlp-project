from typing import Dict, Optional, List
from collections import OrderedDict
import asyncio
import gc

from config.logger import logger
from src.pipeline import RAGPipeline


class LRUPipelineCache:
    """LRU Cache to manage pipelines in memory"""
    
    def __init__(self, max_size: int = 10):
        self.cache: OrderedDict[str, RAGPipeline] = OrderedDict()
        self.max_size = max_size
        self._lock = asyncio.Lock()
        logger.info(f"🗄️ LRUPipelineCache initialized (max_size={max_size})")
    
    async def get(self, service_key: str) -> Optional[RAGPipeline]:
        """Retrieve a pipeline from the cache"""
        async with self._lock:
            if service_key in self.cache:
                # Move to end (mark as most recently used)
                self.cache.move_to_end(service_key)
                logger.debug(f"📦 Pipeline '{service_key}' found in cache")
                return self.cache[service_key]
            return None
    
    async def put(self, service_key: str, pipeline: RAGPipeline):
        """Add a pipeline to the cache with LRU eviction"""
        async with self._lock:
            if service_key in self.cache:
                # Update and move to end
                self.cache.move_to_end(service_key)
                logger.debug(f"📦 Pipeline '{service_key}' updated in cache")
            else:
                self.cache[service_key] = pipeline
                logger.info(f"📦 Pipeline '{service_key}' added to cache ({len(self.cache)}/{self.max_size})")
                
                # Evict if capacity exceeded
                if len(self.cache) > self.max_size:
                    await self._evict_oldest()
    
    async def _evict_oldest(self):
        """Remove the least recently used pipeline"""
        if not self.cache:
            return
            
        evicted_key, evicted_pipeline = self.cache.popitem(last=False)
        logger.info(f"🗑️ Pipeline '{evicted_key}' removed from cache (LRU eviction)")
        
        # Explicit cleanup to free up memory
        await self._cleanup_pipeline(evicted_pipeline)
    
    async def _cleanup_pipeline(self, pipeline: RAGPipeline):
        """Clean up pipeline memory"""
        try:
            if hasattr(pipeline.base_retriever, 'clear_memory'):
                await pipeline.base_retriever.clear_memory()
            
            # Manual cleanup of heavy objects
            if hasattr(pipeline.base_retriever, 'faiss_index'):
                pipeline.base_retriever.faiss_index = None
            if hasattr(pipeline.base_retriever, 'embeddings'):
                pipeline.base_retriever.embeddings = None
            if hasattr(pipeline.base_retriever, 'chunks'):
                pipeline.base_retriever.chunks = []
            if hasattr(pipeline.base_retriever, 'bm25'):
                pipeline.base_retriever.bm25 = None
            
            del pipeline
            
            # Forced garbage collection
            gc.collect()
        except Exception as e:
            logger.error(f"❌ Error during pipeline cleanup: {e}")
    
    async def remove(self, service_key: str) -> bool:
        """Remove a specific pipeline from the cache"""
        async with self._lock:
            if service_key in self.cache:
                pipeline = self.cache.pop(service_key)
                logger.info(f"🗑️ Pipeline '{service_key}' removed from cache")
                await self._cleanup_pipeline(pipeline)
                return True
            return False
    
    async def clear(self):
        """Completely clear the cache"""
        async with self._lock:
            logger.info(f"🗑️ Clearing cache ({len(self.cache)} pipelines)...")
            
            # Explicit cleanup of all pipelines
            for service_key, pipeline in list(self.cache.items()):
                await self._cleanup_pipeline(pipeline)
            
            self.cache.clear()
            gc.collect()
            logger.info("✅ Pipeline cache completely cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache usage statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": f"{len(self.cache)}/{self.max_size}",
            "cached_services": list(self.cache.keys())
        }
    
    async def preload(self, service_keys: List[str], load_func):
        """Pre-load a list of pipelines into the cache"""
        logger.info(f"🔄 Pre-loading {len(service_keys)} pipelines...")
        
        for service_key in service_keys[:self.max_size]:
            pipeline = await load_func(service_key)
            if pipeline:
                await self.put(service_key, pipeline)
        
        logger.info(f"✅ Pre-loading completed ({len(self.cache)} pipelines in cache)")
from typing import List, Dict
from pathlib import Path
import shutil
import time
import asyncio

from config.logger import logger
from config.settings import DOCS_FOLDER_PATH, TOP_K

from schemas.classes import Document
from src.chuncker import semantic_chunking
from src.text_processor import clean_text
from src.utils import load_text_from_file

from src.retrievers.hybrid.hybrid_retriever import HybridRetriever

class RAGPipeline:
    """Complete RAG Pipeline with chunking, indexing, and hybrid search"""
    
    def __init__(self, service_name: str, customer_name: str, retriever=None):
        self.service_name = service_name
        self.customer_name = customer_name
        self.retriever = HybridRetriever(service_name, customer_name) if retriever is None else retriever
        logger.debug(f"{customer_name} - {service_name} - 🚀 FastRAGPipeline initialized for service")
    
    async def add_document_async(
        self, 
        text: str, 
        doc_name: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None
    ) -> bool:
        """Add a document to the pipeline, adapting metadata to the active retriever"""
        logger.debug(f"{self.customer_name} - {self.service_name} - 📄 Processing document: {doc_name}...")
        
        try:
            # 1. Text cleaning
            cleaned_text = clean_text(text)
            
            # 2. Identify retriever and document title
            retriever_name = self.retriever.__class__.__name__
            doc_title = Path(doc_name).stem # Gets file name without extension
            
            # 3. Semantic chunking with conditional logic (via retriever_name)
            chunks = await semantic_chunking(
                cleaned_text,
                doc_name,
                retriever_name=retriever_name,
                doc_title=doc_title,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size
            )
            
            # 5. Save chunks into the retriever's internal list
            self.retriever.chunks.extend(chunks)

            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Created {len(chunks)} chunks for '{doc_name}' (Strategy: {retriever_name})")
            return True
            
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error processing '{doc_name}': {e}", exc_info=True)
            return False
    
    async def build_index_async(self):
        """Build the search index"""
        try:
            success = await self.retriever.index_chunks()
            
            if not success: 
                return False            
            
            logger.debug(f"🔍 ({self.customer_name} - {self.service_name}) Index built successfully")
            return True
        except Exception as e:
            logger.debug(f"❌ ({self.customer_name} - {self.service_name}) Error building index: {e}")
            return False
        
    async def load_documents_batch(
        self, 
        docs_to_upload: List[Document] 
    ) -> List[Document]:
        """Load documents in batch into the RAG pipeline"""
        base_path = Path(DOCS_FOLDER_PATH)
        uploaded_docs: List[Document] = []

        try:
            if not base_path.exists():
                logger.error(f"❌ ({self.customer_name} - {self.service_name}) Directory {DOCS_FOLDER_PATH} not found")
                return []

            for document in docs_to_upload:
                doc_path = base_path / document.name

                if not doc_path.is_file():
                    logger.error(f"❌ ({self.customer_name} - {self.service_name}) Invalid or missing document: {doc_path}")
                    continue
                
                content = await load_text_from_file(str(doc_path))
                if not content.strip():
                    logger.debug(f"⚠️ ({self.customer_name} - {self.service_name}) Document {document.name} is empty or unreadable")
                    continue

                success = await self.add_document_async(
                    text=content,
                    doc_name=document.name,
                    chunk_size=document.chunk_size,
                    chunk_overlap=document.chunk_overlap,
                    min_chunk_size=document.min_chunk_size
                )

                if success:
                    uploaded_docs.append(document)
                    logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Uploaded {document.name}")
                else:
                    logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error uploading {document.name}")

            logger.debug(f"📊 ({self.customer_name} - {self.service_name}) Uploaded {len(uploaded_docs)}/{len(docs_to_upload)} documents")
            return uploaded_docs
        
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error uploading {document.name}: {e}", exc_info=True)
            return []        

    async def search(
        self, 
        query: str,
        top_k: int = TOP_K,
    ) -> Dict[str, Dict]:
        """Search relevant documents using Base, Hybrid V1, and Hybrid V2"""
        logger.debug(f"🔎 ({self.customer_name} - {self.service_name}) Searching query ='{query}'")

        try:
            start = time.time()
            results = await self.retriever.search(
                query=query,
                top_k=top_k
            )
            final_time = time.time() - start
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Search completed with {len(results)} results in {final_time:.4f}s")
            logger.debug(f"({self.customer_name} - {self.service_name}) Results: {results}")
            logger.info(f"⏱️ Retriever - search time: {final_time:.4f}s")
            
            return {
                "results": results,
                "search_time": final_time
            }

        except Exception as e:
            logger.error(
                f"❌ ({self.customer_name} - {self.service_name}) Error during search: {e}",
                exc_info=True
            )
            return None
        
    async def get_context_for_query(
        self, 
        query: str, 
        max_context_length: int = 2000,
        top_k: int = TOP_K
    ) -> str:
        """Get optimized context for response generation (LLM)"""
        logger.debug(f"🔄 ({self.customer_name} - {self.service_name}) Generating context for query: '{query}'")
        context_pieces = []
        current_length = 0
        start_time = time.time()
        
        search_response = await self.search(query, top_k=top_k)
        results = search_response.get("results", []) if search_response else []
        
        for result in results:
            content = result['content']

            if current_length + len(content) <= max_context_length:
                context_pieces.append(f"[Doc: {result['doc_id']}] {content}")
                current_length += len(content)
            else:
                remaining = max_context_length - current_length
                if remaining > 100:
                    context_pieces.append(f"[Doc: {result['doc_id']}] {content[:remaining]}...")
                break
        
        search_time = time.time() - start_time
        logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Context generated with {len(context_pieces)} pieces, total length {current_length} in {search_time:.2f}s")
        return "\n\n".join(context_pieces)
    
    async def save_pipeline(
        self, 
        base_path: str
    ) -> bool:
        """Save pipeline state to disk"""
        try:
            success = await self.retriever.save_index(base_path)

            if not success:
                logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving indices for Retriever")
                return False

            logger.debug(f"💾 ({self.customer_name} - {self.service_name}) Pipeline saved successfully at {base_path}")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving pipeline at {base_path}: {e}", exc_info=True)
            return False

    async def load_pipeline(
        self, 
        base_path: str
    ) -> bool:
        """Load pipeline state from disk"""
        try:
            success = await self.retriever.load_index(base_path)

            if not success:
                logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error loading indices for Retriever")
                return False

            logger.debug(f"📂 ({self.customer_name} - {self.service_name}) Pipeline loaded successfully from {base_path}")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error loading pipeline from {base_path}: {e}", exc_info=True)
            return False
    
    async def delete_pipeline(
        self, 
        base_path: str
    ) -> bool:
        """Delete saved pipeline from disk"""
        target_path = Path(base_path)

        try:
            if not target_path.exists():
                logger.debug(f"⚠️ ({self.customer_name} - {self.service_name}) No pipeline found at {base_path}")
                return False
            
            await asyncio.to_thread(shutil.rmtree, target_path)
            logger.debug(f"🗑️ ({self.customer_name} - {self.service_name}) Pipeline deleted successfully from {base_path}")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error deleting pipeline: {e}", exc_info=True)
            return False
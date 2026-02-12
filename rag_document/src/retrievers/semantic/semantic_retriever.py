from pathlib import Path
from typing import List, Tuple
import faiss  # type: ignore
import numpy as np  # type: ignore
import time
import asyncio

from config.logger import logger
from config.settings import TOP_K

from src.retrievers.i_retriever import IRetriever
from schemas.classes import Chunk
from src.utils import load_compressed_pickle, save_compressed_pickle
from src.embedder import get_embeddings


class SemanticRetriever(IRetriever):
    """Semantic Retriever based on FAISS"""
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        logger.debug(f"✅ ({self.customer_name} - {self.service_name}) SemanticRetriever initialized")

    async def index_chunks(self) -> bool:
        """Creates FAISS indices for the chunks"""
        n_chunks = len(self.chunks)
        logger.debug(f"🔄 ({self.customer_name} - {self.service_name}) Indexing {n_chunks} chunks...")
        start_time = time.time()
        
        try:
            if n_chunks == 0:
                logger.debug(f"⚠️ ({self.customer_name} - {self.service_name}) No chunks to index.")
                return False

            # Embedding Generation
            chunk_texts = [chunk.content for chunk in self.chunks]
            embeddings = await get_embeddings(chunk_texts)
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings.astype('float32'))

            total_time = time.time() - start_time
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Indexing completed in {total_time:.2f}s — {n_chunks} vectors added.")
            return True
        
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error during indexing: {e}")
            return False

    async def search(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> List[Tuple[Chunk, float]]:
        """Semantic search using FAISS"""
        logger.debug(f"🧠 ({self.customer_name} - {self.service_name}) Base semantic search query: {query}")
        
        if self.faiss_index is None:
            logger.error("❌ FAISS index not initialized.")
            return []
        
        try:
            # semantic_search is assumed to be defined in the base class IRetriever
            scores, indices = await self.semantic_search(query, top_k)

            results = [
                (self.chunks[i], float(scores[idx]))
                for idx, i in enumerate(indices)
                if i != -1 and i < len(self.chunks) 
            ]

            return results
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error during search: {e}")
            return []

    async def save_index(self, filepath: str) -> bool:
        """Saves the index to disk"""
        base_path = Path(filepath) / "semantic"
        base_path.mkdir(parents=True, exist_ok=True)
        
        index_path = base_path / "index.pkl.gz"
        faiss_path = base_path / "index.faiss"

        try:
            if self.faiss_index is None:
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"FAISS index not initialized"
                )
                return False
            
            index_data = {'chunks': self.chunks}
            success = await save_compressed_pickle(str(index_path), index_data)

            if not success:
                logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving index metadata")
                return False
            
            # Use running loop to run CPU-bound write_index in executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                faiss.write_index,
                self.faiss_index,
                str(faiss_path)
            )

            logger.debug(f"✅ Index saved to {base_path}")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving index at {filepath}: {e}")
            return False

    async def load_index(self, filepath: str) -> bool:
        """Loads the index from disk"""
        base_path = Path(filepath) / "semantic"
        
        try:
            if not base_path.exists():
                logger.warning(
                    f"⚠️ ({self.customer_name} - {self.service_name}) "
                    f"Path not found: {base_path}"
                )
                return False
            
            index_path = base_path / "index.pkl.gz"
            faiss_path = base_path / "index.faiss"
            
            # Verify file existence
            if not index_path.exists() or not faiss_path.exists():
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"Missing index files in {base_path}"
                )
                return False
            
            # Load chunks metadata
            index_data = await load_compressed_pickle(str(index_path))
            if index_data is None:
                return False
            
            # Load FAISS index in executor
            loop = asyncio.get_running_loop()
            self.faiss_index = await loop.run_in_executor(
                None,
                faiss.read_index,
                str(faiss_path)
            )
            
            self.chunks = index_data['chunks']
            
            logger.debug(
                f"✅ ({self.customer_name} - {self.service_name}) "
                f"Index loaded from {base_path}"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"❌ ({self.customer_name} - {self.service_name}) "
                f"Error loading from {base_path}: {e}"
            )
            return False

    async def clear_memory(self) -> bool:
        """Frees the memory occupied by the retriever"""
        try:
            logger.debug(f"🧹 ({self.customer_name} - {self.service_name}) Clearing retriever memory...")
            
            # Release heavy data structures
            self.chunks = []
            self.faiss_index = None
            
            # Trigger garbage collection
            import gc
            gc.collect()
            
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Retriever memory cleared")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error during memory cleanup: {e}")
            return False
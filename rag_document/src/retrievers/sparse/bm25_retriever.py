from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi  # type: ignore
import faiss  # type: ignore
import numpy as np # type: ignore
import time
import asyncio

from config.logger import logger
from config.settings import TOP_K

from schemas.classes import Chunk
from src.retrievers.i_retriever import IRetriever
from src.text_processor import tokenize_for_bm25
from src.utils import load_compressed_pickle, save_compressed_pickle


class Bm25Retriever(IRetriever):
    """BM25 Retriever"""
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Bm25Retriever initialized")

    async def index_chunks(self) -> bool:
        """Indexes the chunks using BM25"""
        n_chunks = len(self.chunks)
        logger.debug(f"🔄 ({self.customer_name} - {self.service_name}) Indexing {n_chunks} chunks...")
        start_time = time.time()
        
        try:
            if n_chunks == 0:
                logger.debug(f"⚠️ ({self.customer_name} - {self.service_name}) No chunks to index.")
                return False

            chunk_texts = [chunk.content for chunk in self.chunks]

            # Creating index for BM25
            logger.debug(f"⚙️ ({self.customer_name} - {self.service_name}) Tokenizing for BM25...")
            loop = asyncio.get_running_loop()
            bm25_corpus = await loop.run_in_executor(
                None,
                lambda: [tokenize_for_bm25(text) for text in chunk_texts]
            )
            self.bm25 = BM25Okapi(bm25_corpus)

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
        """BM25 search (often used in parallel with semantic search)"""
        logger.debug(f"🔍 ({self.customer_name} - {self.service_name}) Searching query: {query}")

        try:
            # Compute BM25 scores (bm25_search is assumed to be defined in IRetriever)
            bm25_scores = await self.bm25_search(query)
            
            # Normalize scores
            max_score = bm25_scores.max()
            bm25_norm = bm25_scores / (max_score + 1e-6) if max_score > 0 else bm25_scores

            # Get top-K
            actual_top_k = min(top_k, len(bm25_norm))
            if actual_top_k == 0:
                return []
                
            # Perform partitioning to find the top indices efficiently
            top_indices = np.argpartition(-bm25_norm, actual_top_k - 1)[:actual_top_k]
            # Sort top indices by score descending
            top_indices = top_indices[np.argsort(-bm25_norm[top_indices])]

            # Return results
            return [
                (self.chunks[i], float(bm25_norm[i]))
                for i in top_indices
                if i < len(self.chunks)
            ]

        except Exception as e:
            logger.error(
                f"❌ ({self.customer_name} - {self.service_name}) "
                f"Error during search: {e}"
            )
            return []

    async def save_index(self, filepath: str) -> bool:
        """Saves the BM25 index to disk"""
        base_path = Path(filepath) / "bm25"
        base_path.mkdir(parents=True, exist_ok=True)
        
        index_path = base_path / "index.pkl.gz"
        
        try:
            if self.bm25 is None:
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"BM25 index not initialized"
                )
                return False

            chunk_texts = [chunk.content for chunk in self.chunks]
            loop = asyncio.get_running_loop()
            
            # Re-tokenize corpus for persistence if needed, or save existing
            bm25_corpus = await loop.run_in_executor(
                None,
                lambda: [tokenize_for_bm25(text) for text in chunk_texts]
            )

            index_data = {
                'chunks': self.chunks,
                'bm25_corpus': bm25_corpus,
            }

            success = await save_compressed_pickle(str(index_path), index_data)
            
            if not success:
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"Error during saving"
                )
                return False

            logger.debug(f"✅ BM25 Index saved to {base_path}")
            return True
            
        except Exception as e:
            logger.error(
                f"❌ ({self.customer_name} - {self.service_name}) "
                f"Error saving to {base_path}: {e}"
            )
            return False

    async def load_index(self, filepath: str) -> bool:
        """Loads the BM25 index from disk"""
        base_path = Path(filepath) / "bm25"
        
        try:
            if not base_path.exists():
                logger.warning(
                    f"⚠️ ({self.customer_name} - {self.service_name}) "
                    f"Path not found: {base_path}"
                )
                return False

            index_path = base_path / "index.pkl.gz"
            
            if not index_path.exists():
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"Missing index file: {index_path}"
                )
                return False

            index_data = await load_compressed_pickle(str(index_path))
            
            if index_data is None:
                return False
            
            # Load BM25 data
            self.chunks = index_data['chunks']
            self.bm25 = BM25Okapi(index_data['bm25_corpus'])

            logger.debug(
                f"✅ ({self.customer_name} - {self.service_name}) "
                f"BM25 Index loaded from {base_path}"
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
            logger.debug(
                f"🧹 ({self.customer_name} - {self.service_name}) "
                f"Clearing retriever memory..."
            )
            
            self.chunks = []
            self.bm25 = None
            
            import gc
            gc.collect()
            
            logger.debug(
                f"✅ ({self.customer_name} - {self.service_name}) "
                f"Memory cleared"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"❌ ({self.customer_name} - {self.service_name}) "
                f"Error during memory cleanup: {e}"
            )
            return False
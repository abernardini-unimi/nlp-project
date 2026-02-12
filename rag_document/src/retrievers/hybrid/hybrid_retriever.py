from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi  # type: ignore
import faiss  # type: ignore
import numpy as np # type: ignore
import time
import asyncio

from config.logger import logger
from config.settings import TOP_K, BM25_WEIGHT

from schemas.classes import Chunk
from src.retrievers.i_retriever import IRetriever
from src.text_processor import tokenize_for_bm25
from src.utils import load_compressed_pickle, save_compressed_pickle
from src.embedder import get_embeddings


class HybridRetriever(IRetriever):
    """Hybrid Retriever combining BM25 keyword search and Semantic Embeddings"""
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        logger.debug(f"✅ ({self.customer_name} - {self.service_name}) HybridRetriever initialized")

    async def index_chunks(self) -> bool:
        """Index chunks using both BM25 and FAISS"""
        n_chunks = len(self.chunks)
        logger.debug(f"🔄 ({self.customer_name} - {self.service_name}) Indexing {n_chunks} chunks...")
        start_time = time.time()
        
        try:
            if n_chunks == 0:
                logger.debug(f"⚠️ ({self.customer_name} - {self.service_name}) No chunks to index.")
                return False

            chunk_texts = [chunk.content for chunk in self.chunks]

            # 1. BM25 Index Creation
            logger.debug(f"⚙️ ({self.customer_name} - {self.service_name}) Tokenizing for BM25...")
            loop = asyncio.get_event_loop()
            bm25_corpus = await loop.run_in_executor(
                None,
                lambda: [tokenize_for_bm25(text) for text in chunk_texts]
            )
            self.bm25 = BM25Okapi(bm25_corpus)

            # 2. FAISS Index Creation
            embeddings = await get_embeddings(chunk_texts)
            embeddings = embeddings.astype('float32')            
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]            
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings)

            total_time = time.time() - start_time
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Indexing completed in {total_time:.2f}s — {n_chunks} vectors added.")
            return True
        
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Indexing error: {e}")
            return False

    async def search(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> List[Tuple[Chunk, float]]:
        """Hybrid search executing BM25 and Semantic search in parallel"""
        logger.debug(f"🔍 ({self.customer_name} - {self.service_name}) Searching query: {query}")

        try:
            # Parallel execution of both search methods
            bm25_scores, (sem_scores, sem_indices) = await asyncio.gather(
                self.bm25_search(query),
                self.semantic_search(query, len(self.chunks)) # Score all to prevent shape mismatch
            )

            # 1. BM25 Normalization (0-1)
            max_bm25 = bm25_scores.max()
            bm25_norm = bm25_scores / (max_bm25 + 1e-6) if max_bm25 > 0 else bm25_scores

            # 2. Semantic Normalization
            # Map semantic results to the original chunk order
            semantic_all_scores = np.zeros(len(self.chunks))
            for idx, score in zip(sem_indices, sem_scores):
                if idx != -1: 
                    semantic_all_scores[idx] = score
            
            # Min-Max Normalization for semantic scores
            max_sem = semantic_all_scores.max()
            min_sem = semantic_all_scores.min()
            if max_sem > min_sem:
                sem_norm = (semantic_all_scores - min_sem) / (max_sem - min_sem + 1e-6)
            else:
                sem_norm = semantic_all_scores

            # 3. Weighted Combination
            combined_scores = (BM25_WEIGHT * bm25_norm) + ((1 - BM25_WEIGHT) * sem_norm)

            # 4. Top-K Selection
            top_k_val = min(top_k, len(combined_scores))
            top_indices = np.argsort(-combined_scores)[:top_k_val]

            return [(self.chunks[i], float(combined_scores[i])) for i in top_indices]

        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Hybrid search error: {e}")
            return []

    async def save_index(self, filepath: str) -> bool:
        """Save the hybrid index to disk"""
        base_path = Path(filepath) / "hby"
        base_path.mkdir(parents=True, exist_ok=True)
        save_path = str(base_path / "index.pkl.gz")
        
        try:       
            chunk_texts = [chunk.content for chunk in self.chunks]
            loop = asyncio.get_event_loop()
            bm25_corpus = await loop.run_in_executor(
                None,
                lambda: [tokenize_for_bm25(text) for text in chunk_texts]
            )

            index_data = {
                'chunks': self.chunks,
                'bm25_corpus': bm25_corpus,
            }

            success = await save_compressed_pickle(save_path, index_data)    
            if not success:
                logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving index pickle")
                return False
            
            # Save FAISS index
            faiss_path = save_path.replace(".pkl.gz", ".faiss") 
            await loop.run_in_executor(
                None,
                lambda: faiss.write_index(self.faiss_index, faiss_path)
            )

            logger.debug(f"✅ Index successfully saved at {save_path}")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving indices at {filepath}: {e}")
            return False

    async def load_index(self, filepath: str) -> bool:
        """Load the hybrid index from disk"""
        base_path = Path(filepath)
        try:
            if not base_path.exists():
                logger.debug(f"⚠️ ({self.customer_name} - {self.service_name}) No pipeline found at {base_path}")
                return False
            
            index_file = str(base_path / "hby" / "index.pkl.gz")
            index_data = await load_compressed_pickle(index_file)
            
            if index_data is None:
                logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error loading index data")
                return False
            
            # Load FAISS index
            faiss_file = index_file.replace(".pkl.gz", ".faiss")
            loop = asyncio.get_event_loop()
            self.faiss_index = await loop.run_in_executor(
                None,
                lambda: faiss.read_index(faiss_file)
            )
            
            self.chunks = index_data['chunks']
            self.bm25 = BM25Okapi(index_data['bm25_corpus'])

            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Index successfully loaded from {index_file}")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error loading index from {filepath}: {e}")
            return False

    async def clear_memory(self) -> bool:
        """Free memory occupied by the retriever"""
        try:
            logger.debug(f"🧹 ({self.customer_name} - {self.service_name}) Clearing retriever memory...")
            self.chunks = []
            self.bm25 = None
            self.faiss_index = None
            
            import gc
            gc.collect()
            
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Retriever memory cleared")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Memory cleanup error: {e}")
            return False
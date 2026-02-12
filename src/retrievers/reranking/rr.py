from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict
from schemas.classes import Chunk
import faiss  # type: ignore
import asyncio
import numpy as np  # type: ignore

from config.logger import logger
from config.settings import TOP_K

from src.retrievers.i_retriever import IRetriever 
from src.embedder import get_embeddings
from llm.groq import groq_inference
from src.utils import load_compressed_pickle, save_compressed_pickle

# --- Reranking Prompt ---
RERANK_PROMPT = """
You are an expert relevance grader. 
Given a query and a document chunk, assign a score from 0 to 10 based on how relevant the document is to answer the query.
0 = Completely irrelevant
10 = Perfect answer
Respond ONLY with the number.

QUERY: {query}
DOCUMENT: {chunk_text}
SCORE:"""

class RerankingRetriever(IRetriever):
    """
    Reranking Retriever.
    Performs an initial semantic search (Top-20) and then re-orders results 
    using an LLM for surgical precision.
    """
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        self.top_n_initial = 20  # Number of candidates to extract before reranking
        logger.debug(f"✅ ({self.customer_name}) RerankingRetriever initialized")

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

    async def _get_relevance_score(self, query: str, chunk_content: str) -> float:
        """Asks the LLM to evaluate the relevance of a chunk"""
        prompt = RERANK_PROMPT.format(query=query, chunk_text=chunk_content)
        try:
            response = await groq_inference(prompt)
            # Cleaning and conversion to float
            score = float(response.strip())
            return score
        except (ValueError, TypeError):
            return 0.0

    async def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """Semantic search followed by LLM-based reranking"""
        if self.faiss_index is None: 
            logger.error("❌ FAISS index not initialized.")
            return []

        try:
            # 1. Initial Retrieval (Search more than top_k)
            # Obtain the top_n_initial candidates that are vectorially "closest"
            scores, indices = await self.semantic_search(query, top_k=self.top_n_initial)
            
            candidates = []
            for idx in indices:
                if idx != -1 and idx < len(self.chunks):
                    candidates.append(self.chunks[idx])

            if not candidates:
                return []

            # 2. Reranking (Parallelized)
            # Request a relevance score for each candidate
            logger.debug(f"⚖️ Reranking {len(candidates)} candidates...")
            
            ranking_tasks = [self._get_relevance_score(query, c.content) for c in candidates]
            relevance_scores = await asyncio.gather(*ranking_tasks)

            # 3. Sorting based on the new scores
            # Pair chunks with their new scores
            ranked_results = list(zip(candidates, relevance_scores))
            
            # Sort by descending score (10 is better than 0)
            ranked_results.sort(key=lambda x: x[1], reverse=True)

            # 4. Return only the requested top_k results
            return ranked_results[:top_k]
        except Exception as e:
            logger.error(f"❌ ({self.customer_name}) Error search: {e}")
            return False

    async def save_index(self, filepath: str) -> bool:
        """Saves the index to disk"""
        base_path = Path(filepath) / "rrr"
        base_path.mkdir(parents=True, exist_ok=True)
        
        index_path = base_path / "index.pkl.gz"
        faiss_path = base_path / "index.faiss"

        try:
            if self.faiss_index is None:
                logger.error(f"❌ ({self.customer_name}) FAISS index not initialized")
                return False
            
            index_data = {'chunks': self.chunks}
            success = await save_compressed_pickle(str(index_path), index_data)

            if not success:
                logger.error(f"❌ ({self.customer_name}) Error saving index metadata")
                return False
            
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
            logger.error(f"❌ ({self.customer_name}) Error saving indices at {filepath}: {e}")
            return False

    async def load_index(self, filepath: str) -> bool:
        """Loads the index from disk"""
        base_path = Path(filepath) / "rrr"
        
        try:
            if not base_path.exists():
                logger.warning(f"⚠️ ({self.customer_name}) Path not found: {base_path}")
                return False
            
            index_path = base_path / "index.pkl.gz"
            faiss_path = base_path / "index.faiss"
            
            if not index_path.exists() or not faiss_path.exists():
                logger.error(f"❌ ({self.customer_name}) Missing index files in {base_path}")
                return False
            
            # Load chunks
            index_data = await load_compressed_pickle(str(index_path))
            if index_data is None:
                return False
            
            # Load FAISS index
            loop = asyncio.get_running_loop()
            self.faiss_index = await loop.run_in_executor(
                None,
                faiss.read_index,
                str(faiss_path)
            )
            
            self.chunks = index_data['chunks']
            logger.debug(f"✅ ({self.customer_name}) Index loaded from {base_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ ({self.customer_name}) Error loading from {base_path}: {e}")
            return False

    async def clear_memory(self) -> bool:
        """Frees the memory occupied by the retriever"""
        try:
            logger.debug(f"🧹 ({self.customer_name}) Clearing retriever memory...")
            self.chunks = []
            self.faiss_index = None
            
            import gc
            gc.collect()
            
            logger.debug(f"✅ ({self.customer_name}) Memory cleared")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name}) Error during memory cleanup: {e}")
            return False
from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict
from schemas.classes import Chunk
import faiss #type: ignore
import asyncio
import numpy as np #type: ignore

from config.logger import logger
from config.settings import TOP_K

from src.retrievers.i_retriever import IRetriever 
from src.embedder import get_embeddings
from src.text_processor import tokenize_for_bm25
from llm.groq import groq_inference
from src.utils import load_compressed_pickle, save_compressed_pickle

# --- Transformation Prompts ---
STEP_BACK_PROMPT = """
You are an expert at search query optimization. Given a specific user question, 
generate a broader, high-level 'step-back' question that covers the underlying concepts or principles.
Original Question: {query}
Step-back Question:"""

DECOMPOSITION_PROMPT = """
Break down the following complex question into 2 or 3 simpler sub-questions that can be answered independently.
Provide only the questions, one per line.
Complex Question: {query}
Sub-questions:"""

class QueryTransformRetriever(IRetriever):
    """
    Query Transformations Retriever.
    Uses Step-back and Decomposition strategies to generate multiple queries and enrich retrieval.
    """
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        logger.debug(f"✅ ({self.customer_name}) QueryTransformRetriever initialized")


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
            
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index.add(embeddings.astype("float32"))
            faiss.normalize_L2(embeddings)

            total_time = time.time() - start_time
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Indexing completed in {total_time:.2f}s — {n_chunks} vectors added.")
            return True
        
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error during indexing: {e}")
            return False
        

    async def _transform_query(self, query: str) -> List[str]:
        """Generates transformed versions of the query via LLM"""
        # Run Step-back and Decomposition in parallel to optimize latency
        tasks = [
            groq_inference(STEP_BACK_PROMPT.format(query=query)),
            groq_inference(DECOMPOSITION_PROMPT.format(query=query))
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Cleanup and merge results
        transformed_queries = [query] # Always include the original query
        
        step_back = results[0].strip()
        if step_back: transformed_queries.append(step_back)
        
        decomposition = results[1].strip().split('\n')
        transformed_queries.extend([q.strip() for q in decomposition if q.strip()])
        
        logger.debug(f"🔄 Transformed queries: {transformed_queries}")
        return list(set(transformed_queries)) # De-duplication


    async def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """Executes multi-query search and aggregates results"""
        if self.faiss_index is None: return []

        try:
            # 1. Transformation
            all_queries = await self._transform_query(query)

            # 2. Multiple Search (Parallelized)
            # Execute search for each generated query variant
            search_tasks = [self.semantic_search(q, top_k=5) for q in all_queries]
            search_results = await asyncio.gather(*search_tasks)

            # 3. Result Aggregation (Simplified Reciprocal Rank Fusion)
            chunk_scores: Dict[int, float] = {}
            
            for scores, indices in search_results:
                for score, idx in zip(scores, indices):
                    if idx == -1: continue
                    # Using an inverse weighted sum here to penalize distant duplicates.
                    chunk_scores[idx] = chunk_scores.get(idx, 0) + (1.0 / (score + 1.0))

            # 4. Final sorting
            sorted_indices = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
            
            final_chunks = []
            for idx, score in sorted_indices[:top_k]:
                final_chunks.append((self.chunks[idx], float(score)))

            return final_chunks
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error search: {e}")
            return []


    async def save_index(self, filepath: str) -> bool:
        """Saves the index to disk"""
        base_path = Path(filepath) / "qt"
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
                logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving indices")
                return False
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                faiss.write_index,
                self.faiss_index,
                str(faiss_path)
            )

            logger.debug(f"✅ Index saved to {base_path}")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving indices at {filepath}: {e}")
            return False


    async def load_index(self, filepath: str) -> bool:
        """Loads the index from disk"""
        base_path = Path(filepath) / "qt"
        
        try:
            if not base_path.exists():
                logger.warning(
                    f"⚠️ ({self.customer_name} - {self.service_name}) "
                    f"Path not found: {base_path}"
                )
                return False
            
            index_path = base_path / "index.pkl.gz"
            faiss_path = base_path / "index.faiss"
            
            # Check for file existence
            if not index_path.exists() or not faiss_path.exists():
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"Missing index files in {base_path}"
                )
                return False
            
            # Load chunks
            index_data = await load_compressed_pickle(str(index_path))
            if index_data is None:
                return False
            
            # Load FAISS index
            loop = asyncio.get_event_loop()
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
        """Frees memory occupied by the retriever"""
        try:
            logger.debug(f"🧹 ({self.customer_name} - {self.service_name}) Clearing retriever memory...")
            
            # Release heavy data structures
            self.chunks = []
            self.faiss_index = None
            
            # Forced garbage collection
            import gc
            gc.collect()
            
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Retriever memory cleared")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error during memory cleanup: {e}")
            return False
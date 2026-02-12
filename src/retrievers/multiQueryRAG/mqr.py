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

# --- Multi-Query Prompt ---
MULTI_QUERY_PROMPT = """
You are an AI language model assistant. Your task is to generate 3 
different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {query}
"""

class MultiQueryRetriever(IRetriever):
    """
    Multi-Query RAG Retriever.
    Generates query variations to cover different areas of the semantic space.
    """
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        logger.debug(f"✅ ({self.customer_name}) MultiQueryRetriever initialized")

    async def index_chunks(self) -> bool:
        """Creates FAISS indices for chunks"""
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

    async def _generate_variants(self, query: str) -> List[str]:
        """Generates query variations using Groq"""
        response = await groq_inference(MULTI_QUERY_PROMPT.format(query=query))
        
        # Cleanup: include the original query + generated variants
        lines = response.strip().split('\n')
        variants = [query] + [line.strip() for line in lines if line.strip()]
        
        # Limit to the first 4 total queries to prevent latency spikes
        return list(set(variants))[:4]

    async def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        if self.faiss_index is None: return []

        # 1. Variant generation
        queries = await self._generate_variants(query)
        logger.debug(f"🔍 Running search for {len(queries)} variants...")

        # 2. Parallel semantic searches
        search_tasks = [self.semantic_search(q, top_k=top_k) for q in queries]
        results_list = await asyncio.gather(*search_tasks)

        # 3. Result aggregation (Merging and De-duplication)
        # Using a dictionary to track the BEST score for each chunk.
        unique_chunks: Dict[int, float] = {}
        
        for scores, indices in results_list:
            for score, idx in zip(scores, indices):
                if idx == -1: continue
                
                # For IndexFlatIP, a HIGHER score means higher similarity.
                # We keep the maximum score found for each chunk across all query variants.
                if idx not in unique_chunks or score > unique_chunks[idx]:
                    unique_chunks[idx] = score

        # 4. Final sorting
        # reverse=True because higher scores (similarity) should come first.
        sorted_indices = sorted(unique_chunks.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for idx, score in sorted_indices[:top_k]:
            final_results.append((self.chunks[idx], float(score)))

        return final_results

    async def save_index(self, filepath: str) -> bool:
        """Saves the index to disk"""
        base_path = Path(filepath) / "mqr"
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
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error saving indices in {filepath}: {e}")
            return False

    async def load_index(self, filepath: str) -> bool:
        """Loads the index from disk"""
        base_path = Path(filepath) / "mqr"
        
        try:
            if not base_path.exists():
                logger.warning(
                    f"⚠️ ({self.customer_name} - {self.service_name}) "
                    f"Path not found: {base_path}"
                )
                return False
            
            index_path = base_path / "index.pkl.gz"
            faiss_path = base_path / "index.faiss"
            
            # Check file existence
            if not index_path.exists() or not faiss_path.exists():
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"Index files missing in {base_path}"
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
        """Frees the memory occupied by the retriever"""
        try:
            logger.debug(f"🧹 ({self.customer_name} - {self.service_name}) Cleaning up retriever memory...")
            
            # Release heavy data structures
            self.chunks = []
            self.faiss_index = None
            
            # Garbage collection
            import gc
            gc.collect()
            
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Retriever memory cleared")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error during memory cleanup: {e}")
            return False
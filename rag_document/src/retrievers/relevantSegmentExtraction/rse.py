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
from src.utils import load_compressed_pickle, save_compressed_pickle

class RelevantSegmentRetriever(IRetriever):
    """
    Relevant Segment Extraction (RSE) Retriever.
    Identifies clusters of contiguous chunks that maximize overall relevance. 
    """
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        self.threshold = 0.4  # Minimum score to consider a chunk "relevant"
        self.max_gap = 1       # Number of "weak" chunks to skip when joining two segments
        logger.debug(f"✅ ({self.customer_name}) RelevantSegmentRetriever initialized")

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

    async def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """
        RSE Algorithm: 
        1. Compute scores for ALL chunks.
        2. Identify segments (contiguous sequences).
        3. Return chunks from the top-scoring segments.
        """
        if self.faiss_index is None: return []

        try:
            # 1. Obtain semantic scores for all chunks
            scores, indices = await self.semantic_search(query, len(self.chunks))
            
            flat_scores = scores[0] if len(scores.shape) > 1 else scores
            flat_indices = indices[0] if len(indices.shape) > 1 else indices

            # Map chunk_id -> score
            chunk_scores = {int(idx): float(score) for score, idx in zip(flat_scores, flat_indices) if idx != -1}

            # 2. Group chunks by document and sort by index
            doc_groups: Dict[str, List[int]] = {}
            for i, chunk in enumerate(self.chunks):
                doc_id = chunk.doc_id
                if doc_id not in doc_groups: doc_groups[doc_id] = []
                doc_groups[doc_id].append(i)

            all_segments = []

            # 3. Segment Extraction (RSE Logic)
            for doc_id, chunk_indices in doc_groups.items():
                # Sort by position within the document
                chunk_indices.sort(key=lambda x: self.chunks[x].chunk_index)
                
                current_segment = []
                gap_count = 0

                for idx in chunk_indices:
                    score = chunk_scores.get(idx, 0)
                    
                    if score >= self.threshold:
                        current_segment.append((idx, score))
                        gap_count = 0  # Reset gap if a strong chunk is found
                    else:
                        if current_segment and gap_count < self.max_gap:
                            current_segment.append((idx, score))
                            gap_count += 1
                        else:
                            if current_segment:
                                # Save segment if valid (e.g., high average score)
                                avg_score = sum(s for _, s in current_segment) / len(current_segment)
                                all_segments.append((current_segment, avg_score))
                            current_segment = []
                            gap_count = 0

                if current_segment:
                    avg_score = sum(s for _, s in current_segment) / len(current_segment)
                    all_segments.append((current_segment, avg_score))
            
            # 4. Result selection and flattening
            # Sort segments by average score and take the best ones
            all_segments.sort(key=lambda x: x[1], reverse=True)

            if not all_segments:
                return []
            
            best_segment_chunks = all_segments[0][0][:top_k]
            
            final_results: List[Tuple[Chunk, float]] = []
            for chunk_idx, score in best_segment_chunks[:top_k]:
                final_results.append((self.chunks[chunk_idx], float(score)))
                
            return final_results
        
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error search: {e}")
            return []

    async def save_index(self, filepath: str) -> bool:
        """Saves the index to disk"""
        base_path = Path(filepath) / "rse"
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
        base_path = Path(filepath) / "rse"
        
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
        """Frees the memory occupied by the retriever"""
        try:
            logger.debug(f"🧹 ({self.customer_name} - {self.service_name}) Clearing retriever memory...")
            
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
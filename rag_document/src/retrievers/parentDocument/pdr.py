from pathlib import Path
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

class ParentDocumentRetriever(IRetriever):
    """
    Parent Document Retrieval (Small-to-Big).
    Searches across 'Child Chunks' (small) but returns 'Parent Chunks' (big).
    """
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        # Dictionary to map parent ID to its full text
        self.parent_store: Dict[str, str] = {}
        logger.debug(f"✅ ({self.customer_name}) ParentDocumentRetriever initialized")

    async def index_chunks(self) -> bool:
        """
        Indexes Child Chunks and populates the parent_store.
        Assumes self.chunks contains 'Child' chunks with 'Parent' references in metadata.
        """
        if not self.chunks:
            return False

        logger.info(f"🚀 ({self.customer_name}) Starting Small-to-Big indexing...")

        # 1. Populate parent_store for fast retrieval
        for chunk in self.chunks:
            p_id = chunk.metadata.get("parent_id")
            p_text = chunk.metadata.get("parent_text")
            if p_id and p_text and p_id not in self.parent_store:
                self.parent_store[p_id] = p_text

        # 2. Generate Embeddings for Child Chunks (high precision)
        child_texts = [c.content for c in self.chunks]
        embeddings = await get_embeddings(child_texts)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        
        # 3. FAISS Index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

        return True

    async def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """
        Performs search on children but returns parent documents.
        """
        if self.faiss_index is None:
            return []

        # 1. Semantic search to find the best Child Chunks
        # We retrieve more results (top_k * 2) to account for parent de-duplication
        scores, indices = await self.semantic_search(query, top_k * 2) 
        
        seen_parents = set()
        final_results: List[Tuple[Chunk, float]] = []

        for score, idx in zip(scores, indices):
            if idx == -1: continue
            
            child_chunk = self.chunks[idx]
            parent_id = child_chunk.metadata.get("parent_id")

            # Check if this parent has already been added to the results
            if parent_id and parent_id not in seen_parents:
                parent_text = self.parent_store.get(parent_id)
                
                if parent_text:
                    # Create a new Chunk object representing the Parent
                    parent_chunk = Chunk(
                        id=f"parent_{parent_id}",
                        content=parent_text,
                        doc_id=child_chunk.doc_id,
                        start_pos=child_chunk.start_pos,
                        end_pos=child_chunk.end_pos,
                        chunk_index=child_chunk.chunk_index,
                        metadata=child_chunk.metadata.copy()
                    )
                    parent_chunk.metadata["is_parent"] = True
                    
                    final_results.append((parent_chunk, float(score)))
                    seen_parents.add(parent_id)
            
            # Stop once the required top_k unique documents are reached
            if len(final_results) >= top_k:
                break

        # Note: final_results is already sorted by semantic search scores (descending)
        return final_results

    async def save_index(self, filepath: str) -> bool:
        """Saves the index to disk"""
        base_path = Path(filepath) / "pdr"
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
            
            # Store chunks and the parent mapping
            index_data = {
                'chunks': self.chunks,
                'parent_store': self.parent_store
            }
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
        base_path = Path(filepath) / "pdr"
        
        try:
            if not base_path.exists():
                logger.warning(
                    f"⚠️ ({self.customer_name} - {self.service_name}) "
                    f"Path not found: {base_path}"
                )
                return False
            
            index_path = base_path / "index.pkl.gz"
            faiss_path = base_path / "index.faiss"
            
            if not index_path.exists() or not faiss_path.exists():
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"Missing index files in {base_path}"
                )
                return False
            
            # Load chunks and parent_store
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
            self.parent_store = index_data.get('parent_store', {})
            
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
            
            self.chunks = []
            self.faiss_index = None
            self.parent_store = {}
            
            import gc
            gc.collect()
            
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Retriever memory cleared")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Error during memory cleanup: {e}")
            return False
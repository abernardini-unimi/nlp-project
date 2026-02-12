from abc import ABC, abstractmethod
import asyncio
from typing import List, Tuple, Optional
import faiss # type: ignore
import numpy as np  # type: ignore
from schemas.classes import Chunk
from src.embedder import get_embeddings
from src.text_processor import tokenize_for_bm25


class IRetriever(ABC):
    """Interface for all Retriever classes"""
    
    def __init__(self, service_name: str, customer_name: str):
        self.service_name = service_name                # Service Name
        self.customer_name = customer_name              # Customer Name
        self.chunks: List[Chunk] = []                   # List of chunks
        self.faiss_index: Optional[object] = None       # FAISS index
        self.bm25: Optional[object] = None              # BM25 index/settings
        self.embeddings: Optional[np.ndarray] = None    # Chunk embeddings
        self.retriever: Optional[object] = None         # Only for Chroma DB

    @abstractmethod
    async def index_chunks(self) -> bool:
        """
        Index the chunks in the vector store.
        Takes the chunks in self.chunks and indexes them (FAISS, BM25, or both).
        Returns a boolean indicating if indexing succeeded.
        """
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        """
        Main search function in the vector store.
        Searches the query using the previously created index and retrieves the corresponding chunks.
        Returns a list of chunks with their scores.
        """
        pass

    @abstractmethod
    async def save_index(self, filepath: str) -> bool:
        """
        Save necessary structures to disk.
        Stores the indices and chunks in the specified folder.
        Returns a boolean indicating success.
        """
        pass

    @abstractmethod
    async def load_index(self, filepath: str) -> bool:
        """
        Load necessary structures from disk.
        Retrieves all information previously saved for the retriever.
        Returns a boolean indicating success.
        """
        pass

    @abstractmethod
    async def clear_memory(self) -> bool:
        """
        Free memory and reset the retriever state.
        Deletes all data structures created for the retriever.
        Returns a boolean indicating success.
        """
        pass

    async def semantic_search(
        self,
        query: str,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the semantic score of a query using the FAISS index."""
        if self.faiss_index is None:
            raise ValueError("FAISS index is not initialized")
        
        loop = asyncio.get_event_loop()

        query_embedding = await get_embeddings([query])
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = await loop.run_in_executor(
            None,
            lambda: self.faiss_index.search(query_embedding, top_k)
        )

        return scores[0], indices[0]

    async def bm25_search(
        self,
        query: str
    ) -> np.ndarray:
        """Compute BM25 scores for a query using the BM25 index."""
        
        if self.bm25 is None:
            raise ValueError("BM25 index is not initialized")
        
        loop = asyncio.get_event_loop()
        bm25_tokens = tokenize_for_bm25(query)

        bm25_scores = await loop.run_in_executor(
            None,
            lambda: np.array(self.bm25.get_scores(bm25_tokens), dtype='float32')
        )

        return bm25_scores

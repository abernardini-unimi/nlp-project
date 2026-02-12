from pathlib import Path
from typing import List, Tuple, Optional, Dict
from schemas.classes import Chunk
import faiss #type: ignore
import asyncio
import numpy as np #type: ignore
import tiktoken #type: ignore

from config.logger import logger
from config.settings import TOP_K

from src.retrievers.i_retriever import IRetriever 
from llm.groq import groq_inference
from src.utils import load_compressed_pickle, save_compressed_pickle
from src.embedder import get_embeddings

DOCUMENT_TITLE_PROMPT = """
INSTRUCTIONS
What is the title of the following document?

Your response MUST be the title of the document, and nothing else. DO NOT respond with anything else.

DOCUMENT
{document_text}
""".strip()

TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')
MAX_CONTENT_TOKENS = 4000

class ContextualHeaderRetriever(IRetriever):
    """
    Contextual Chunk Headers retriever. Enriches each chunk with a contextual header before indexing.
    """
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        logger.debug(f"✅ ({self.customer_name} - {self.service_name}) ContextualHeaderRetriever initialized")


    def _truncate_content(self, content: str, max_tokens: int) -> str:
        tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
        return TOKEN_ENCODER.decode(tokens[:max_tokens])


    async def _get_document_title(self, document_text: str) -> str:
        """Extracts the document title using groq_inference"""
        try:
            truncated_text = self._truncate_content(document_text, MAX_CONTENT_TOKENS)
            query = DOCUMENT_TITLE_PROMPT.format(document_text=truncated_text)
            
            title = await groq_inference(query)
            return title.strip() if title else "Untitled Document"
        
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Get document Title Error: {e}")
            return "Untitled Document"


    async def index_chunks(self) -> bool:
        """
        Enriches chunks with the document title and creates FAISS and BM25 indices.
        """
        try: 
            if not self.chunks:
                logger.warning(f"⚠️ ({self.customer_name}) No chunks to index.")
                return False

            logger.info(f"🚀 ({self.customer_name}) Starting contextual enrichment of {len(self.chunks)} chunks...")

            # 1. Document -> Title mapping (to avoid duplicate calls)
            doc_titles: Dict[str, str] = {}
            unique_docs: Dict[str, str] = {}

            for chunk in self.chunks:
                doc_id = chunk.metadata.get("document_title", "unknown")
                if doc_id not in unique_docs:
                    # Use doc_text if present, otherwise the content of the first chunk found
                    unique_docs[doc_id] = chunk.metadata.get("doc_text", chunk.content)

            # 2. Asynchronous title extraction
            for doc_id, text in unique_docs.items():
                doc_titles[doc_id] = await self._get_document_title(text)

            # 3. Creation of enriched texts
            enriched_texts = []
            for chunk in self.chunks:
                doc_id = chunk.metadata.get("document_title", "unknown")
                title = doc_titles.get(doc_id, "Untitled Document")
                
                # Formatting the contextual header
                enriched_content = f"Document Title: {title}\n\nContent: {chunk.content}"
                enriched_texts.append(enriched_content)
                
                # Update chunk content for future generation
                chunk.content = enriched_content

            # 4. Embeddings generation and FAISS indexing
            logger.debug(f"🧠 ({self.customer_name}) Generating embeddings...")
            embeddings = await get_embeddings(enriched_texts) 

            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings.astype("float32"))

            logger.info(f"✅ ({self.customer_name}) Indexing completed with Contextual Headers.")
            return True
        
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Indexing error: {e}")
            return False


    async def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """Semantic search on enriched chunks"""
        if self.faiss_index is None:
            logger.error("❌ FAISS index not initialized.")
            return []
        
        try:
            scores, indices = await self.semantic_search(query, top_k)

            results = [
                (self.chunks[i], float(scores[idx]))
                for idx, i in enumerate(indices)
                if i != -1 and i < len(self.chunks) 
            ]

            return results
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Errore durante la ricerca: {e}")
            return []

    
    async def save_index(self, filepath: str) -> bool:
        """Saves the index to disk"""
        base_path = Path(filepath) / "chr"
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
        base_path = Path(filepath) / "chr"
        
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
            
            # Free heavy data structures
            self.chunks = []
            self.faiss_index = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Retriever memory cleared")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Memory cleanup error: {e}")
            return False
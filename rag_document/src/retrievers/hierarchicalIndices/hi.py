from pathlib import Path
from typing import List, Tuple, Optional, Dict
from schemas.classes import Chunk
import faiss #type: ignore
import asyncio
import numpy as np #type: ignore

from config.logger import logger
from config.settings import TOP_K

from src.retrievers.i_retriever import IRetriever 
from src.embedder import get_embeddings
from llm.groq import groq_inference
from src.utils import load_compressed_pickle, save_compressed_pickle

SUMMARY_PROMPT = """
Summarize the following document content to highlight its main topics and key information. 
The summary will be used for vector search.
DOCUMENT CONTENT:
{text}
"""

class HierarchicalRetriever(IRetriever):
    """
    Hierarchical Indices Retriever.
    Usa un indice di alto livello (riassunti) per filtrare i documenti 
    e un indice di basso livello per i chunk specifici.
    """
    
    def __init__(self, service_name: str, customer_name: str):
        super().__init__(service_name, customer_name)
        self.summary_index: Optional[faiss.IndexFlatIP] = None
        self.doc_ids_list: List[str] = [] # Mappa gli indici di summary_index agli ID doc
        logger.debug(f"✅ ({self.customer_name}) HierarchicalRetriever inizializzato")

    async def _generate_summary(self, text: str) -> str:
        prompt = SUMMARY_PROMPT.format(text=text[:6000]) # Limitiamo il testo per il riassunto
        summary = await groq_inference(prompt)
        return summary.strip() if summary else ""

    async def index_chunks(self) -> bool:
        if not self.chunks:
            return False

        logger.info(f"🚀 ({self.customer_name}) Creazione Gerarchia Indici...")

        # 1. Raggruppa chunk per documento
        doc_map: Dict[str, List[Chunk]] = {}
        for chunk in self.chunks:
            d_id = chunk.metadata.get("document_id", "default")
            if d_id not in doc_map: doc_map[d_id] = []
            doc_map[d_id].append(chunk)

        # 2. Genera riassunti per ogni documento (Top Level)
        summaries = []
        self.doc_ids_list = []
        for d_id, chunks in doc_map.items():
            full_text = " ".join([c.content for c in chunks[:5]]) # Prime parti del doc
            summary = await self._generate_summary(full_text)
            summaries.append(summary)
            self.doc_ids_list.append(d_id)

        # 3. Indicizza i Riassunti (Top Level Index)
        summary_embeddings = await get_embeddings(summaries)
        self.summary_index = faiss.IndexFlatIP(summary_embeddings.shape[1])
        self.summary_index.add(summary_embeddings.astype("float32"))

        # 4. Indicizza i Chunk normali (Bottom Level Index)
        self.embeddings = await get_embeddings([c.content for c in self.chunks])
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings.astype("float32"))

        return True

    async def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """
        Ricerca a due stadi:
        1. Trova i documenti più rilevanti tramite i riassunti.
        2. Cerca i chunk solo dentro quei documenti.
        """
        if self.summary_index is None or self.faiss_index is None:
            return []

        # --- Stadio 1: Ricerca sui riassunti ---
        query_embedding = await get_embeddings([query])
        s_scores, s_indices = self.summary_index.search(query_embedding.astype("float32"), k=3)
        
        relevant_doc_ids = [self.doc_ids_list[i] for i in s_indices[0] if i != -1]
        logger.debug(f"🔍 Documenti filtrati: {relevant_doc_ids}")

        # --- Stadio 2: Ricerca sui chunk con filtraggio ---
        # Recuperiamo più chunk e poi filtriamo per document_id
        c_scores, c_indices = self.faiss_index.search(query_embedding.astype("float32"), k=top_k * 5)
        
        final_results = []
        for score, idx in zip(c_scores[0], c_indices[0]):
            if idx == -1: continue
            chunk = self.chunks[idx]
            
            # Filtro: teniamo il chunk solo se appartiene ai documenti top
            if chunk.metadata.get("document_id") in relevant_doc_ids:
                final_results.append((chunk, float(score)))
            
            if len(final_results) >= top_k:
                break

        return final_results

    async def save_index(self, filepath: str) -> bool:
        """Salva l'indice su disco"""
        base_path = Path(filepath) / "faiss_base"
        base_path.mkdir(parents=True, exist_ok=True)
        
        index_path = base_path / "index.pkl.gz"
        faiss_path = base_path / "index.faiss"

        try:
            if self.faiss_index is None:
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"Indice FAISS non inizializzato"
                )
                return False
            
            index_data = {'chunks': self.chunks}
            success = await save_compressed_pickle(str(index_path), index_data)

            if not success:
                logger.error(f"❌ ({self.customer_name} - {self.service_name}) Errore nel salvataggio degli indici")
                return False
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                faiss.write_index,
                self.faiss_index,
                str(faiss_path)
            )

            logger.debug(f"✅ Indice salvato in {base_path}")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Errore nel salvataggio degli indici in {filepath}: {e}")
            return False


    async def load_index(self, filepath: str) -> bool:
        """Carica l'indice da disco"""
        base_path = Path(filepath) / "faiss_base"
        
        try:
            if not base_path.exists():
                logger.warning(
                    f"⚠️ ({self.customer_name} - {self.service_name}) "
                    f"Path non trovato: {base_path}"
                )
                return False
            
            index_path = base_path / "index.pkl.gz"
            faiss_path = base_path / "index.faiss"
            
            # Verifica esistenza file
            if not index_path.exists() or not faiss_path.exists():
                logger.error(
                    f"❌ ({self.customer_name} - {self.service_name}) "
                    f"File indice mancanti in {base_path}"
                )
                return False
            
            # Carica chunks
            index_data = await load_compressed_pickle(str(index_path))
            if index_data is None:
                return False
            
            # Carica FAISS index
            loop = asyncio.get_event_loop()
            self.faiss_index = await loop.run_in_executor(
                None,
                faiss.read_index,
                str(faiss_path)
            )
            
            self.chunks = index_data['chunks']
            
            logger.debug(
                f"✅ ({self.customer_name} - {self.service_name}) "
                f"Indice caricato da {base_path}"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"❌ ({self.customer_name} - {self.service_name}) "
                f"Errore nel caricamento da {base_path}: {e}"
            )
            return False
        

    async def clear_memory(self) -> bool:
        """Libera la memoria occupata dal retriever"""
        try:
            logger.debug(f"🧹 ({self.customer_name} - {self.service_name}) Pulizia memoria retriever...")
            
            # Libera strutture dati pesanti
            self.chunks = []
            self.faiss_index = None
            self.summary_index = None
            self.doc_ids_list = None
            
            # Garbage collection
            import gc
            gc.collect()
            
            logger.debug(f"✅ ({self.customer_name} - {self.service_name}) Memoria retriever liberata")
            return True
        except Exception as e:
            logger.error(f"❌ ({self.customer_name} - {self.service_name}) Errore pulizia memoria: {e}")
            return False
from typing import List
import torch # type: ignore
import numpy as np # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import asyncio

from config.logger import logger
from config.settings import EMBEDDING_MODEL_NAME

# Global model
_embedding_model = None
_embedding_model_lock = asyncio.Lock()

async def get_embedding_model():
    """Upload the model once and share it across all pipelines"""
    global _embedding_model
    if _embedding_model is None:
        async with _embedding_model_lock:
            if _embedding_model is None:  # Double-check
                loop = asyncio.get_event_loop()
                _embedding_model = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(EMBEDDING_MODEL_NAME)
                )
                logger.info(f"🤖 Modello embedding caricato: {EMBEDDING_MODEL_NAME}")
    return _embedding_model


async def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generates embeddings for a list of texts."""
    batch_size = 256 if torch.cuda.is_available() else 64
    
    model = await get_embedding_model()
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
    )
    return embeddings


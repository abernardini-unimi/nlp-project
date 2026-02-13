import asyncio
from typing import Optional
from threading import Lock  # <-- Importa Lock

from config.logger import logger
from src.service_manager import MultiServiceManager

_manager_instance: Optional[MultiServiceManager] = None
_creation_lock = Lock()

def get_manager(retriever_type: str = "SemanticRetriever") -> MultiServiceManager:
    """Returns the singleton instance of the MultiServiceManager"""
    global _manager_instance
    
    if _manager_instance is None:
        with _creation_lock:
            if _manager_instance is None:
                _manager_instance = MultiServiceManager(retriever_type=retriever_type)
    
    return _manager_instance


def reset_manager() -> None:
    """Reset manager singleton"""
    global _manager_instance
    _manager_instance = None
    logger.warning("♻️ Manager singleton reset")
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from pydantic import BaseModel  # type: ignore
import numpy as np  # type: ignore

class Chunk(BaseModel):
    """Text chunk with extended metadata for advanced RAG techniques"""
    id: str
    content: str
    doc_id: str
    start_pos: int
    end_pos: int
    chunk_index: int                    # For RSE (maintains order)
    parent_id: Optional[str] = None     # For Parent Document Retrieval
    parent_text: Optional[str] = None   # Full text or parent paragraph
    metadata: Dict[str, Any] = {}       # For Contextual Headers (title, doc_text, etc.)

class Document(BaseModel):
    """Representation of a document"""
    name: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int

class Service(BaseModel):
    """Structured configuration for a service"""
    name: str
    description: str 
    tokens: List[str]
    customer_name: str
    documents: List[Document] = []

    def __hash__(self):
        """Hash based on name and customer_name"""
        return hash((self.name, self.customer_name))
    
    def __eq__(self, other):
        """Equality based on name and customer_name"""
        if not isinstance(other, Service):
            return False
        return self.name == other.name and self.customer_name == other.customer_name

    def to_dict(self) -> Dict:
        """Convert the Service instance to a dictionary"""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Service':
        """Create a Service instance from a dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

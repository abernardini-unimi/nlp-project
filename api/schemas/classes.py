from typing import List
from pydantic import BaseModel # type: ignore
from schemas.classes import Document

class SaveServicePayload(BaseModel):
    name: str
    description: str 
    tokens: List[str]
    customer_name: str
    documents: List[Document] = []

class DeleteServicePayload(BaseModel):
    service_name: str
    customer_name: str

class RunTestPayload(BaseModel):
    num_queries: int

# Pydantic schemas
from pydantic import BaseModel

class ResearchRequest(BaseModel):
    topic: str

class SearchRequest(BaseModel):
    query: str


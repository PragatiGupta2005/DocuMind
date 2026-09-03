from typing import Any
from pydantic import BaseModel, Field

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
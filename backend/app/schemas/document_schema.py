from pydantic import BaseModel
from typing import Dict, Optional


class DocumentSchema(BaseModel):
    filename: str
    file_type: str
    text: str
    metadata: Dict[str, Optional[str]]
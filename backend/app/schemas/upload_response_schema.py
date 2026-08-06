from pydantic import BaseModel
from app.schemas.document_schema import DocumentSchema

class UploadResponseSchema(BaseModel):
    filename: str
    original_filename: str
    storage_path: str
    content_type: str
    size: int
    message: str
    document: DocumentSchema
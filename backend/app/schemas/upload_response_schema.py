from pydantic import BaseModel
from app.schemas.document_schema import DocumentSchema
from app.schemas.chunk_schema import ChunkSchema

class UploadResponseSchema(BaseModel):
    """
    Response returned after a document has been
    uploaded, processed, and chunked.
    """

    filename: str
    original_filename: str
    storage_path: str
    content_type: str
    size: int
    message: str
    document: DocumentSchema
    chunks: list[ChunkSchema]
    chunk_count: int
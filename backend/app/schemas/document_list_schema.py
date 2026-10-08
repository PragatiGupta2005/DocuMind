from pydantic import BaseModel


class DocumentListItemSchema(BaseModel):
    """
    Lightweight representation of a document
    used by the document listing endpoint.
    """

    document_id: str
    filename: str
    file_type: str

    original_filename: str | None = None
    file_size: int | None = None
    created_at: str | None = None
    chunk_count: int | None = None
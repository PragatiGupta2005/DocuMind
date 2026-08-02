from pydantic import BaseModel

class UploadResponse(BaseModel):
    filename: str
    original_filename: str
    size: int
    content_type: str
    storage_path: str
    message: str
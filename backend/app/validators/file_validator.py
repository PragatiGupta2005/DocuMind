from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.constants.file_constants import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
)

class FileValidator:
    @staticmethod
    def validate_extension(file: UploadFile):
        extension = Path(file.filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported extension: {extension}"
            )
    @staticmethod
    def validate_content_type(file: UploadFile):
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported MIME Type: {file.content_type}"
            )

from fastapi import UploadFile

from app.storage.local_storage import LocalStorage
from app.utils.filename_utils import generate_filename


class UploadService:
    """
    Handles the business logic for document uploads.
    """

    def __init__(self):
        self.storage = LocalStorage()

    async def upload_file(self, file: UploadFile):

        unique_filename = generate_filename(file.filename)

        storage_path = await self.storage.save(
            file=file,
            filename=unique_filename
        )

        return {
            "filename": unique_filename,
            "original_filename": file.filename,
            "size": file.size,
            "content_type": file.content_type,
            "storage_path": storage_path,
            "message": "File uploaded successfully"
        }
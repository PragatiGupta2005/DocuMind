from fastapi import UploadFile

from app.storage.local_storage import LocalStorage
from app.utils.filename_utils import generate_filename

from app.services.document_processing_service import DocumentProcessingService

from app.schemas.upload_response_schema import UploadResponseSchema


class UploadService:
    """
    Handles the business logic for document uploads.
    """

    def __init__(self):
        self.storage = LocalStorage()
        self.processing_service = DocumentProcessingService()

    async def upload_file(self, file: UploadFile):

        # Generate a unique filename
        unique_filename = generate_filename(file.filename)

        # Save the uploaded file
        storage_path = await self.storage.save(
            file=file,
            filename=unique_filename
        )

        # Process the saved document
        document = self.processing_service.process_document(
            storage_path
        )

        # Return upload response
        return UploadResponseSchema(
            filename=unique_filename,
            original_filename=file.filename,
            size=file.size,
            content_type=file.content_type,
            storage_path=storage_path,
            message="File uploaded successfully",
            document=document
        )
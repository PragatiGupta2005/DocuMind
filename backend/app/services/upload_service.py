from fastapi import UploadFile
from app.storage.local_storage import LocalStorage
from app.utils.filename_utils import generate_filename
from app.services.document_processing_service import (DocumentProcessingService)
from app.services.chunking_service import (ChunkingService)
from app.schemas.upload_response_schema import (UploadResponseSchema)

class UploadService:
    """
    Handles the business logic for document uploads.
    """

    def __init__(self):
        self.storage = LocalStorage()
        self.processing_service = (DocumentProcessingService())
        self.chunking_service = (ChunkingService())

    async def upload_file(self,file: UploadFile):

        # Step 1: Generate unique filename
        unique_filename = generate_filename(file.filename)

        # Step 2: Save uploaded file
        storage_path = await self.storage.save(file=file,filename=unique_filename)

        # Step 3: Process document
        document = (self.processing_service.process_document(storage_path))

        # Step 4: Chunk document
        chunks = (self.chunking_service.chunk_document(document))

        # Step 5: Return complete response
        return UploadResponseSchema(
            filename=unique_filename,
            original_filename=file.filename,
            size=file.size,
            content_type=file.content_type,
            storage_path=storage_path,
            message="File uploaded, processed, and chunked successfully",
            document=document,
            chunks=chunks,
            chunk_count=len(chunks)
        )
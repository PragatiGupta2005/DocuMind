from fastapi import UploadFile
from app.storage.local_storage import LocalStorage
from app.utils.filename_utils import generate_filename
from app.services.document_processing_service import (DocumentProcessingService)
from app.services.chunking_service import (ChunkingService)
from app.schemas.upload_response_schema import (UploadResponseSchema)
from app.embeddings.embedding_service import EmbeddingService
from app.schemas.vector_store_schema import VectorStoreSchema
from app.vector_store.qdrant_store import QdrantVectorStore
from app.vector_store.collection_config import LOCAL_COLLECTION_NAME

class UploadService:
    """
    Handles the business logic for document uploads.
    """

    def __init__(self):
        self.storage = LocalStorage()
        self.processing_service = DocumentProcessingService()
        self.chunking_service = ChunkingService()
        self.embedding_service = EmbeddingService()
        self.vector_store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    async def upload_file(self,file: UploadFile):

        # Step 1: Generate unique filename
        unique_filename = generate_filename(file.filename)

        # Step 2: Save uploaded file
        storage_path = await self.storage.save(file=file,filename=unique_filename)

        # Step 3: Process document
        document = (self.processing_service.process_document(storage_path))

        # Step 4: Chunk document
        chunks = (self.chunking_service.chunk_document(document))

        # Step 5: Generate embeddings
        embeddings = self.embedding_service.embed_chunks(chunks)

        # Step 6: Convert embeddings to vector records
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append(
                VectorStoreSchema(
                    point_id=chunk.chunk_uuid,
                    document_id=embedding.document_id,
                    chunk_uuid=embedding.chunk_uuid,
                    model_name=embedding.model_name,
                    dimensions=embedding.dimensions,
                    vector=embedding.vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_name": chunk.document_name,
                        "text": chunk.text,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                    },
                )
            )
        self.vector_store.add(records)

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
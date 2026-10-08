from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.upload_service import UploadService
from app.storage.document_registry import DocumentRegistry
from app.services.document_service import DocumentService
from app.schemas.document_list_schema import DocumentListItemSchema

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload, process, and chunk a document.
    """
    upload_service = UploadService()
    return await upload_service.upload_file(
        file
    )

@router.get(
    "",
    response_model=list[DocumentListItemSchema]
)
async def list_documents():

    registry = DocumentRegistry()
    documents = registry.list_all()

    return [
        DocumentListItemSchema(
            document_id=document.document_id,
            filename=document.filename,
            file_type=document.file_type,
            original_filename=document.metadata.get(
                "original_filename"
            ),
            file_size=document.metadata.get(
                "file_size"
            ),
            created_at=document.metadata.get(
                "created_at"
            ),
            chunk_count=document.metadata.get(
                "chunk_count"
            ),
        )
        for document in documents
    ]

@router.get("/{document_id}")
async def get_document(
    document_id: str,
):
    """
    Return a document by its document ID.
    """
    registry = DocumentRegistry()
    document = registry.get(document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )
    return document

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
):
    """
    Delete a document and its associated vectors.
    """

    document_service = DocumentService()

    deleted = document_service.delete_document(
        document_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
    }
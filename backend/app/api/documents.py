from fastapi import APIRouter, File, UploadFile
from app.services.upload_service import UploadService
from app.storage.document_registry import DocumentRegistry

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

@router.get("")
async def list_documents():
    """
    Return all uploaded documents.
    """
    registry = DocumentRegistry()
    return registry.list_all()
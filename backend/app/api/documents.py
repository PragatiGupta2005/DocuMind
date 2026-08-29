from fastapi import APIRouter, File, UploadFile

from app.services.upload_service import UploadService


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
from fastapi import APIRouter, File, UploadFile

from app.controllers.upload_controller import UploadController

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)

controller = UploadController()


@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a single document.
    """
    return await controller.upload(file)
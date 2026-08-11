from fastapi import APIRouter, File, UploadFile
from app.controllers.upload_controller import UploadController
from app.schemas.upload_response_schema import UploadResponseSchema

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)

controller = UploadController()

@router.post(
    "/",
    response_model=UploadResponseSchema,
    summary="Upload Document",
    description="Upload a document, save it, process it, and return the extracted information."
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a single document.
    """
    return await controller.upload(file)
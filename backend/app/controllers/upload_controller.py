from fastapi import UploadFile

from app.services.upload_service import UploadService
from app.validators.file_validator import FileValidator


class UploadController:
    """
    Coordinates validation and upload workflow.
    """

    def __init__(self):
        self.service = UploadService()

    async def upload(self, file: UploadFile):

        FileValidator.validate_extension(file)
        FileValidator.validate_content_type(file)

        return await self.service.upload_file(file)
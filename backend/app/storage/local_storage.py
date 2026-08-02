from pathlib import Path
from fastapi import UploadFile
from app.constants.file_constants import UPLOAD_DIRECTORY

class LocalStorage:
    def __init__(self):
        Path(UPLOAD_DIRECTORY).mkdir(
            exist_ok=True
        )
    async def save(
        self,
        file: UploadFile,
        filename: str
    ):
        path = Path(UPLOAD_DIRECTORY) / filename
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        return str(path)
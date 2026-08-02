import uuid
from pathlib import Path

def generate_filename(filename: str):
    extension = Path(filename).suffix
    return f"{uuid.uuid4()}{extension}"
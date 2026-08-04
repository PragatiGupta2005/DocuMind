import os

from app.processors.pdf_processor import PDFProcessor
from app.schemas.document_schema import DocumentSchema


class DocumentProcessingService:
    """
    Selects the correct processor
    based on the uploaded document type.
    """

    def __init__(self):

        self.pdf_processor = PDFProcessor()

    def process_document(self, file_path: str) -> DocumentSchema:

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":

            return self.pdf_processor.process(file_path)

        raise ValueError(
            f"Unsupported document type: {extension}"
        )
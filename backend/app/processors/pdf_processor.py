import os
import fitz

from app.processors.base_processor import BaseProcessor
from app.schemas.document_schema import DocumentSchema


class PDFProcessor(BaseProcessor):
    """
    PDF document processor.
    Responsible for extracting text and metadata from PDF files.
    """

    def open_document(self, file_path: str):

        return fitz.open(file_path)

    def extract_text(self, document) -> str:

        text = ""

        for page in document:
            text += page.get_text()

        return text

    def extract_metadata(self, document) -> dict:

        metadata = document.metadata

        return {
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
            "creator": metadata.get("creator"),
            "pages": str(document.page_count)
        }

    def clean_text(self, text: str) -> str:

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        return "\n".join(lines)

    def process(self, file_path: str) -> DocumentSchema:

        document = self.open_document(file_path)

        text = self.extract_text(document)

        metadata = self.extract_metadata(document)

        cleaned_text = self.clean_text(text)

        result = DocumentSchema(

            filename=os.path.basename(file_path),

            file_type="pdf",

            text=cleaned_text,

            metadata=metadata

        )

        document.close()

        return result
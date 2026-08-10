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
        """
        Open the PDF document using PyMuPDF.
        """
        return fitz.open(file_path)

    def extract_text(self, document) -> str:
        """
        Extract text from all pages of the PDF.
        """

        text = ""

        for page in document:
            text += page.get_text()

        return text

    def extract_metadata(self, document) -> dict:
        """
        Extract metadata from the PDF.
        """

        metadata = document.metadata

        return {
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
            "creator": metadata.get("creator"),
            "pages": str(document.page_count)
        }

    def clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text.
        Removes unnecessary whitespace and blank lines.
        """

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        return "\n".join(lines)

    def process(self, file_path: str) -> DocumentSchema:
        """
        Complete PDF processing pipeline.

        Steps:
        1. Open PDF
        2. Extract text
        3. Extract metadata
        4. Clean text
        5. Generate document ID
        6. Return DocumentSchema
        """

        document = self.open_document(file_path)

        try:
            text = self.extract_text(document)
            metadata = self.extract_metadata(document)
            cleaned_text = self.clean_text(text)

            document_id = os.path.splitext(
                os.path.basename(file_path)
            )[0]

            # Step 5: Create DocumentSchema
            result = DocumentSchema(

                document_id=document_id,
                filename=os.path.basename(file_path),
                file_type="pdf",
                text=cleaned_text,
                metadata=metadata

            )
            return result

        finally:
            document.close()
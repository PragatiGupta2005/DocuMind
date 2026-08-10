import os

from docx import Document

from app.processors.base_processor import BaseProcessor
from app.schemas.document_schema import DocumentSchema


class DOCXProcessor(BaseProcessor):
    """
    Processor responsible for extracting text and metadata
    from Microsoft Word (.docx) documents.
    """

    def open_document(self, file_path: str):
        """
        Open the DOCX document.
        """
        return Document(file_path)

    def extract_text(self, document) -> str:
        """
        Extract text from all paragraphs in the DOCX document.
        """

        paragraphs = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                paragraphs.append(
                    paragraph.text.strip()
                )

        return "\n".join(paragraphs)

    def extract_metadata(self, document) -> dict:
        """
        Extract metadata from the DOCX document.
        """

        properties = document.core_properties

        return {

            "title": properties.title,

            "author": properties.author,

            "subject": properties.subject,

            "category": properties.category,

            "created": str(properties.created),

            "modified": str(properties.modified)

        }

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing
        unnecessary whitespace and blank lines.
        """

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:

                lines.append(line)

        return "\n".join(lines)

    def process(self, file_path: str) -> DocumentSchema:
        """
        Complete DOCX processing pipeline.

        Steps:
        1. Open DOCX document
        2. Extract text
        3. Extract metadata
        4. Clean text
        5. Generate document ID
        6. Return DocumentSchema
        """

        document = self.open_document(file_path)

        # Extract text
        text = self.extract_text(document)

        # Extract metadata
        metadata = self.extract_metadata(document)

        # Clean extracted text
        cleaned_text = self.clean_text(text)

        document_id = os.path.splitext(
            os.path.basename(file_path)
        )[0]

        # Create DocumentSchema
        result = DocumentSchema(

            document_id=document_id,

            filename=os.path.basename(file_path),

            file_type="docx",

            text=cleaned_text,

            metadata=metadata

        )

        return result
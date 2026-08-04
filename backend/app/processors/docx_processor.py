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

        return Document(file_path)

    def extract_text(self, document) -> str:

        paragraphs = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                paragraphs.append(paragraph.text.strip())

        return "\n".join(paragraphs)

    def extract_metadata(self, document) -> dict:

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

        return DocumentSchema(

            filename=os.path.basename(file_path),

            file_type="docx",

            text=cleaned_text,

            metadata=metadata

        )
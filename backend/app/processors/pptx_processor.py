import os
from pptx import Presentation

from app.processors.base_processor import BaseProcessor
from app.schemas.document_schema import DocumentSchema


class PPTXProcessor(BaseProcessor):
    """
    Processor responsible for extracting text and metadata
    from Microsoft PowerPoint (.pptx) presentations.
    """

    def open_document(self, file_path: str):

        return Presentation(file_path)

    def extract_text(self, presentation) -> str:

        slides_text = []

        for slide in presentation.slides:

            for shape in slide.shapes:

                if hasattr(shape, "text"):

                    text = shape.text.strip()

                    if text:

                        slides_text.append(text)

        return "\n".join(slides_text)

    def extract_metadata(self, presentation) -> dict:

        properties = presentation.core_properties

        return {

            "title": properties.title,

            "author": properties.author,

            "subject": properties.subject,

            "category": properties.category,

            "created": str(properties.created),

            "modified": str(properties.modified),

            "slides": str(len(presentation.slides))

        }

    def clean_text(self, text: str) -> str:

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:

                lines.append(line)

        return "\n".join(lines)

    def process(self, file_path: str) -> DocumentSchema:

        presentation = self.open_document(file_path)

        text = self.extract_text(presentation)

        metadata = self.extract_metadata(presentation)

        cleaned_text = self.clean_text(text)

        return DocumentSchema(

            filename=os.path.basename(file_path),

            file_type="pptx",

            text=cleaned_text,

            metadata=metadata

        )
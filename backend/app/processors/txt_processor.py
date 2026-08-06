import os

from app.processors.base_processor import BaseProcessor
from app.schemas.document_schema import DocumentSchema


class TXTProcessor(BaseProcessor):
    """
    Processor responsible for extracting text
    from plain text (.txt) documents.
    """

    def open_document(self, file_path: str):

        with open(file_path, "r", encoding="utf-8") as file:

            return file.read()

    def extract_text(self, document: str) -> str:

        return document

    def extract_metadata(self, file_path: str) -> dict:

        stat = os.stat(file_path)

        return {

            "size_bytes": str(stat.st_size),

            "created": str(stat.st_ctime),

            "modified": str(stat.st_mtime)

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

        metadata = self.extract_metadata(file_path)

        cleaned_text = self.clean_text(text)

        return DocumentSchema(

            filename=os.path.basename(file_path),

            file_type="txt",

            text=cleaned_text,

            metadata=metadata

        )
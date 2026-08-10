import os

from app.processors.base_processor import BaseProcessor
from app.schemas.document_schema import DocumentSchema


class TXTProcessor(BaseProcessor):
    """
    Processor responsible for extracting text
    from plain text (.txt) documents.
    """

    def open_document(self, file_path: str):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    def extract_text(self, document: str) -> str:
        """
        Return the contents of the TXT document.
        """

        return document

    def extract_metadata(self, file_path: str) -> dict:
        """
        Extract filesystem metadata from the TXT file.
        """

        stat = os.stat(file_path)

        return {

            "size_bytes": str(stat.st_size),

            "created": str(stat.st_ctime),

            "modified": str(stat.st_mtime)

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
        Complete TXT processing pipeline.

        Steps:
        1. Open TXT file
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
        metadata = self.extract_metadata(file_path)

        # Clean text
        cleaned_text = self.clean_text(text)

        # Generate unique document ID
        #
        # Example:
        # uploads/
        # 69759dba-d439-4cb2-8cd0-af880a4731b9.txt
        #
        # becomes:
        # 69759dba-d439-4cb2-8cd0-af880a4731b9

        document_id = os.path.splitext(
            os.path.basename(file_path)
        )[0]

        # Create DocumentSchema
        result = DocumentSchema(

            document_id=document_id,

            filename=os.path.basename(file_path),

            file_type="txt",

            text=cleaned_text,

            metadata=metadata

        )

        return result
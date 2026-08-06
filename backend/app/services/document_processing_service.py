from pathlib import Path

from app.exceptions.document_exceptions import (
    UnsupportedDocumentTypeError,
)

from app.processors.processor_factory import ProcessorFactory
from app.schemas.document_schema import DocumentSchema


class DocumentProcessingService:
    """
    Coordinates document processing.
    """

    def process_document(
        self,
        file_path: str
    ) -> DocumentSchema:

        extension = Path(file_path).suffix.lower()

        try:

            processor = ProcessorFactory.get_processor(
                extension
            )

            return processor.process(file_path)

        except UnsupportedDocumentTypeError:

            raise
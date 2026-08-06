from app.exceptions.document_exceptions import (
    UnsupportedDocumentTypeError,
)

from app.processors.pdf_processor import PDFProcessor
from app.processors.docx_processor import DOCXProcessor
from app.processors.pptx_processor import PPTXProcessor
from app.processors.txt_processor import TXTProcessor


class ProcessorFactory:
    """
    Factory responsible for returning
    the correct processor.
    """

    _processors = {

        ".pdf": PDFProcessor,

        ".docx": DOCXProcessor,

        ".pptx": PPTXProcessor,

        ".txt": TXTProcessor

    }

    @classmethod
    def get_processor(cls, extension: str):

        extension = extension.lower()

        processor = cls._processors.get(extension)

        if processor is None:

            raise UnsupportedDocumentTypeError(extension)

        return processor()
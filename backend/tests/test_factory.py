import pytest

from app.processors.processor_factory import ProcessorFactory
from app.exceptions.document_exceptions import (
    UnsupportedDocumentTypeError
)


def test_pdf_processor():
    processor = ProcessorFactory.get_processor(".pdf")

    assert processor.__class__.__name__ == "PDFProcessor"


def test_docx_processor():
    processor = ProcessorFactory.get_processor(".docx")

    assert processor.__class__.__name__ == "DOCXProcessor"


def test_pptx_processor():
    processor = ProcessorFactory.get_processor(".pptx")

    assert processor.__class__.__name__ == "PPTXProcessor"


def test_txt_processor():
    processor = ProcessorFactory.get_processor(".txt")

    assert processor.__class__.__name__ == "TXTProcessor"


def test_unsupported_file():
    with pytest.raises(UnsupportedDocumentTypeError):
        ProcessorFactory.get_processor(".exe")
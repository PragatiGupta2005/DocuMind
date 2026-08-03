from abc import ABC, abstractmethod

from app.schemas.document_schema import DocumentSchema


class BaseProcessor(ABC):
    """
    Abstract base class for all document processors.
    """

    @abstractmethod
    def process(self, file_path: str) -> DocumentSchema:
        """
        Read the document and return extracted content.
        """
        pass
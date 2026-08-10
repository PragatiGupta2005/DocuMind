import uuid
from langchain_text_splitters import (RecursiveCharacterTextSplitter)
from app.chunking.base_chunker import BaseChunker
from app.schemas.chunk_schema import ChunkSchema
from app.schemas.document_schema import DocumentSchema

from app.core.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

class RecursiveChunker(BaseChunker):
    """
    Recursive text chunking strategy.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],

            add_start_index=True
        )

    def chunk(
        self,
        document: DocumentSchema
    ) -> list[ChunkSchema]:

        split_documents = self.splitter.create_documents(
            [document.text]
        )

        chunks = []

        for index, split_document in enumerate(
            split_documents,
            start=1
        ):

            text = split_document.page_content

            start_index = split_document.metadata[
                "start_index"
            ]

            end_index = start_index + len(text)

            chunk = ChunkSchema(
                document_id=document.document_id,
                chunk_uuid=str(uuid.uuid4()),
                chunk_id=index,
                document_name=document.filename,
                text=text,
                start_index=start_index,
                end_index=end_index,
                metadata=document.metadata.copy()
            )
            chunks.append(chunk)
        return chunks
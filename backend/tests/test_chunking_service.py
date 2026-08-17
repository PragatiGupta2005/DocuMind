from app.services.chunking_service import ChunkingService
from app.schemas.document_schema import DocumentSchema

document = DocumentSchema(
    document_id="doc_001",
    filename="AI.pdf",
    file_type="pdf",
    text="""
Artificial Intelligence is a branch of computer science.
Machine Learning is a subset of Artificial Intelligence.
Deep Learning uses neural networks.
Natural Language Processing allows computers to understand human language.
Computer Vision allows machines to understand images and videos.
""",

    metadata={
        "title": "Artificial Intelligence"
    }
)
service = ChunkingService()
chunks = service.chunk_document(document)
for chunk in chunks:
    print(chunk.model_dump_json(indent=4))
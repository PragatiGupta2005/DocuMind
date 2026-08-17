from app.services.document_processing_service import (
    DocumentProcessingService
)

service = DocumentProcessingService()

result = service.process_document(
    "uploads/69759dba-d439-4cb2-8cd0-af880a4731b9.pdf"
)

print(result.json(indent=4))
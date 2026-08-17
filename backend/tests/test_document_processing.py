from app.services.document_processing_service import (
    DocumentProcessingService
)

service = DocumentProcessingService()

result = service.process_document(
    "uploads/78b81ce3-fb3c-4bd9-b91c-523e2a2527a4.pdf"
)

print(result.model_dump_json(indent=4))
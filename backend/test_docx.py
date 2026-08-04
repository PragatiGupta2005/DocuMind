from app.processors.docx_processor import DOCXProcessor

processor = DOCXProcessor()

result = processor.process(
    "uploads/f5c34c2b-e0b0-467c-b08a-86f42497450d.docx"
)

print(result.json(indent=4))
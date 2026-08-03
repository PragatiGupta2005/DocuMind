from app.processors.pdf_processor import PDFProcessor

processor = PDFProcessor()

result = processor.process("uploads/69759dba-d439-4cb2-8cd0-af880a4731b9.pdf")

print(result.json(indent=4))
from app.processors.pptx_processor import PPTXProcessor

processor = PPTXProcessor()

result = processor.process(
    "uploads/your_file.pptx"
)

print(result.model_dump_json(indent=4))
from app.processors.pptx_processor import PPTXProcessor

processor = PPTXProcessor()

result = processor.process(
    "e1c94042-146d-4d2c-bad7-03c7719eefec.pptx"
)

print(result.model_dump_json(indent=4))
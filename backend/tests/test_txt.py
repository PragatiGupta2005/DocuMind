from app.processors.txt_processor import TXTProcessor

processor = TXTProcessor()

result = processor.process(
    "uploads/30b678c4-c88e-4f9f-8349-fda108ac1f95.txt"
)

print(result.json(indent=4))
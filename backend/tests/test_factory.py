from app.processors.processor_factory import ProcessorFactory

processor = ProcessorFactory.get_processor(".pdf")

print(type(processor))
print(type(ProcessorFactory.get_processor(".pdf")))
print(type(ProcessorFactory.get_processor(".docx")))
print(type(ProcessorFactory.get_processor(".pptx")))
print(type(ProcessorFactory.get_processor(".txt")))
print(type(ProcessorFactory.get_processor(".exe")))
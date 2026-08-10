from app.chunking.recursive_chunker import RecursiveChunker
from app.schemas.document_schema import DocumentSchema

document = DocumentSchema(
    document_id="doc_001",
    filename="AI.pdf",
    file_type="pdf",
    text = """
Artificial Intelligence is a branch of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence. These tasks include reasoning, learning, problem solving, perception, and language understanding.
Machine Learning is a subset of Artificial Intelligence. It enables computers to learn patterns from data without being explicitly programmed for every task. Machine learning algorithms can be divided into supervised learning, unsupervised learning, and reinforcement learning.
Deep Learning is a specialized area of Machine Learning that uses artificial neural networks with multiple layers. Deep learning has achieved excellent results in computer vision, natural language processing, speech recognition, and many other applications.
Natural Language Processing is another important area of Artificial Intelligence. It allows computers to understand, process, and generate human language. Modern NLP systems use transformer architectures and large language models to perform complex language tasks.
Computer Vision enables machines to interpret and understand visual information from images and videos. Applications include facial recognition, object detection, medical image analysis, autonomous vehicles, and industrial inspection.
Artificial Intelligence continues to evolve rapidly and is being applied across healthcare, finance, education, transportation, cybersecurity, manufacturing, and many other industries.
""",
    metadata={}
)

chunker = RecursiveChunker()

chunks = chunker.chunk(document)

for chunk in chunks:
    print(chunk.model_dump_json(indent=4))
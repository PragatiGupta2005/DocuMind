from app.chunking.chunk_factory import ChunkFactory
chunker = ChunkFactory.get_chunker("recursive")
print(type(chunker))
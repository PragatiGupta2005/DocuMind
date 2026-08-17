from app.schemas.chunk_schema import ChunkSchema

chunk = ChunkSchema(

    chunk_id=1,
    document_name="AI.pdf",
    text="Artificial Intelligence is...",
    start_index=0,
    end_index=120,
    metadata={
        "page": 1
    }

)

print(chunk.json(indent=4))
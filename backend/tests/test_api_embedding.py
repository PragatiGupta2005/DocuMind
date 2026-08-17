from app.embeddings.api_embedding import APIEmbedding


embedding = APIEmbedding()


print("===================================")
print("API Embedding Provider Test")
print("===================================")

print(
    "Model:",
    embedding.get_model_name()
)


# -----------------------------------
# Single embedding
# -----------------------------------

text = (
    "Machine Learning is a subset "
    "of Artificial Intelligence."
)

vector = embedding.embed(text)


print(
    "Vector Type:",
    type(vector)
)

print(
    "Vector Dimensions:",
    len(vector)
)

print(
    "First 5 Values:",
    vector[:5]
)


# -----------------------------------
# Batch embedding
# -----------------------------------

texts = [
    "Artificial Intelligence is a field of computer science.",
    "Machine Learning allows systems to learn from data.",
    "Deep Learning uses neural networks."
]

vectors = embedding.embed_batch(texts)


print(
    "Number of Input Texts:",
    len(texts)
)

print(
    "Number of Generated Vectors:",
    len(vectors)
)

print(
    "Batch Vector Dimensions:",
    len(vectors[0])
)


print("===================================")
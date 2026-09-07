from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model():
    """Create and return the local embedding model."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":
    embeddings = create_embedding_model()

    test_text = "TechNova AI develops artificial intelligence solutions."

    vector = embeddings.embed_query(test_text)

    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
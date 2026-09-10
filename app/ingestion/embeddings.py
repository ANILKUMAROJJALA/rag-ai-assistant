from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL


def create_embedding_model():
    """
    Create and return the local embedding model.

    The model name is loaded from the
    centralized application configuration.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings


if __name__ == "__main__":
    embeddings = create_embedding_model()

    test_text = (
        "TechNova AI develops "
        "artificial intelligence solutions."
    )

    vector = embeddings.embed_query(
        test_text
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Embedding dimensions: "
        f"{len(vector)}"
    )

    print(
        f"First 5 values: "
        f"{vector[:5]}"
    )
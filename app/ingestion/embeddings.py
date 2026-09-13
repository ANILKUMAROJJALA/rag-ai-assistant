from functools import lru_cache

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from app.config import (
    EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def create_embedding_model():
    """
    Create and cache the local embedding model.

    The embedding model is expensive to initialize,
    so one instance is reused for the lifetime of
    the backend process.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


if __name__ == "__main__":

    embeddings = (
        create_embedding_model()
    )

    test_text = (
        "TechNova AI develops "
        "artificial intelligence solutions."
    )

    vector = (
        embeddings.embed_query(
            test_text
        )
    )

    print(
        "Embedding model:",
        EMBEDDING_MODEL,
    )

    print(
        "Embedding dimensions:",
        len(vector),
    )

    print(
        "First 5 values:",
        vector[:5],
    )
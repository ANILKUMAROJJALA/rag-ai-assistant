from functools import lru_cache

from langchain_chroma import (
    Chroma,
)

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
)

from app.ingestion.chunking import (
    split_documents,
)

from app.ingestion.embeddings import (
    create_embedding_model,
)

from app.ingestion.loader import (
    load_documents_from_folder,
)


@lru_cache(maxsize=1)
def create_vector_store():
    """
    Create or open the configured persistent
    Chroma collection.

    The vector store is cached so the backend
    reuses the same Chroma connection.
    """

    return Chroma(
        collection_name=(
            CHROMA_COLLECTION_NAME
        ),
        persist_directory=(
            CHROMA_PERSIST_DIRECTORY
        ),
        embedding_function=(
            create_embedding_model()
        ),
    )


def ingest_documents():
    """
    Development/demo helper.

    Load all supported documents from the
    configured demo folder structure used
    during development and add their chunks
    to Chroma.

    The production upload API does not depend
    on this function.
    """

    print(
        "\nLoading documents..."
    )

    documents = (
        load_documents_from_folder(
            "data/raw"
        )
    )

    chunks = split_documents(
        documents
    )

    if not chunks:
        print(
            "No chunks found."
        )
        return

    ids = [
        chunk.metadata[
            "chunk_id"
        ]
        for chunk in chunks
    ]

    vector_store = (
        create_vector_store()
    )

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    stored = (
        vector_store.get()
    )

    print(
        "\nVector store updated "
        "successfully!"
    )

    print(
        "Chunks processed:",
        len(chunks),
    )

    print(
        "Total records stored:",
        len(
            stored["ids"]
        ),
    )


if __name__ == "__main__":
    ingest_documents()
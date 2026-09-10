from functools import lru_cache

from langchain_chroma import Chroma

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    VECTOR_TOP_K,
)

from app.ingestion.embeddings import (
    create_embedding_model,
)


@lru_cache(maxsize=1)
def create_vector_store():
    """
    Create the Chroma vector store once
    and reuse the same instance.
    """

    embeddings = (
        create_embedding_model()
    )

    return Chroma(
        collection_name=
            CHROMA_COLLECTION_NAME,

        persist_directory=
            CHROMA_PERSIST_DIRECTORY,

        embedding_function=
            embeddings,
    )


def get_available_sources():
    """
    Dynamically retrieve all unique
    source filenames stored in Chroma.
    """

    vector_store = (
        create_vector_store()
    )

    data = vector_store.get(
        include=["metadatas"]
    )

    sources = set()

    for metadata in data["metadatas"]:

        source = metadata.get(
            "source"
        )

        if source:
            sources.add(
                source
            )

    return sorted(
        sources
    )


def retrieve_documents(
    query,
    metadata_filter=None,
    k=None,
):
    """
    Perform semantic retrieval with
    optional metadata filtering.
    """

    if k is None:
        k = VECTOR_TOP_K

    vector_store = (
        create_vector_store()
    )

    if metadata_filter:

        documents = (
            vector_store.similarity_search(
                query,
                k=k,
                filter=metadata_filter,
            )
        )

    else:

        documents = (
            vector_store.similarity_search(
                query,
                k=k,
            )
        )

    return documents
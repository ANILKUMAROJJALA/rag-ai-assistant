from functools import lru_cache

from langchain_chroma import Chroma

from app.ingestion.embeddings import create_embedding_model


PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "technova_documents"


@lru_cache(maxsize=1)
def create_vector_store():
    """
    Create the Chroma vector store once and reuse it.

    lru_cache prevents us from repeatedly loading the embedding model
    every time retrieval or metadata inspection is performed.
    """

    embeddings = create_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def get_available_sources():
    """
    Read all unique document source names dynamically
    from metadata stored inside Chroma.
    """

    vector_store = create_vector_store()

    data = vector_store.get(
        include=["metadatas"]
    )

    sources = set()

    for metadata in data["metadatas"]:
        source = metadata.get("source")

        if source:
            sources.add(source)

    return sorted(sources)


def retrieve_documents(
    query,
    metadata_filter=None,
    k=5,
):
    """
    Perform semantic retrieval with an optional metadata filter.
    """

    vector_store = create_vector_store()

    if metadata_filter:
        documents = vector_store.similarity_search(
            query,
            k=k,
            filter=metadata_filter,
        )

    else:
        documents = vector_store.similarity_search(
            query,
            k=k,
        )

    return documents
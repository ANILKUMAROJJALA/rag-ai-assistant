from langchain_chroma import Chroma

from app.ingestion.loader import (
    load_documents_from_folder,
)

from app.ingestion.chunking import (
    split_documents,
)

from app.ingestion.embeddings import (
    create_embedding_model,
)


PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "technova_documents"


def create_vector_store():
    """
    Create or open the persistent Chroma collection.
    """

    embeddings = (
        create_embedding_model()
    )

    vector_store = Chroma(
        collection_name=
            COLLECTION_NAME,

        persist_directory=
            PERSIST_DIRECTORY,

        embedding_function=
            embeddings,
    )

    return vector_store


def ingest_documents():
    """
    Load documents, split them into chunks,
    and upsert them into Chroma using
    deterministic chunk IDs.
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


    # --------------------------------------------------
    # Upsert instead of blind insertion
    # --------------------------------------------------

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )


    stored = (
        vector_store.get()
    )


    print(
        "\nVector store updated successfully!"
    )

    print(
        "Chunks processed:",
        len(chunks),
    )

    print(
        "Total records stored:",
        len(stored["ids"]),
    )


if __name__ == "__main__":

    ingest_documents()
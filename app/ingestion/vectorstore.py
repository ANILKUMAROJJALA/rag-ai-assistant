from langchain_chroma import Chroma

from app.ingestion.chunking import split_documents
from app.ingestion.embeddings import create_embedding_model
from app.ingestion.loader import load_documents_from_folder


PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "technova_documents"


def create_vector_store():
    """Create a Chroma vector store from all supported documents."""

    documents = load_documents_from_folder(
        "data/raw"
    )

    chunks = split_documents(documents)

    embeddings = create_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


if __name__ == "__main__":

    vector_store = create_vector_store()

    print("\nVector store created successfully!")
    print(
        "Documents stored:",
        vector_store._collection.count(),
    )
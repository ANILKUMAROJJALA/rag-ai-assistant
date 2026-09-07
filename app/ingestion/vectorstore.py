from langchain_chroma import Chroma

from chunking import split_documents
from embeddings import create_embedding_model
from loader import load_text_file


def create_vector_store():
    """Create a Chroma vector store from document chunks."""

    documents = load_text_file("data/raw/company_info.txt")
    chunks = split_documents(documents)

    embeddings = create_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="technova_documents",
    )

    return vector_store


if __name__ == "__main__":
    vector_store = create_vector_store()

    print("Vector store created successfully!")
    print("Documents stored:", vector_store._collection.count())
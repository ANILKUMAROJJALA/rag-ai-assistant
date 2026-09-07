from langchain_chroma import Chroma

from app.ingestion.embeddings import create_embedding_model


def create_retriever():
    """Load the existing Chroma vector store and create a retriever."""

    embeddings = create_embedding_model()

    vector_store = Chroma(
        collection_name="technova_documents",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    return retriever


if __name__ == "__main__":
    retriever = create_retriever()

    query = "Where is TechNova AI headquartered?"

    results = retriever.invoke(query)

    print(f"Query: {query}")
    print(f"\nRetrieved {len(results)} chunks:\n")

    for i, document in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(document.page_content)
        print()
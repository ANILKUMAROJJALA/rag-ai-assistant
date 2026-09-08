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
    search_kwargs={"k": 5}
)


    return retriever


if __name__ == "__main__":
    retriever = create_retriever()

    test_queries = [
        "What does NovaSearch do?",
        "What is TechNova AI's refund policy?",
        "How does TechNova AI protect customer passwords?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"Query: {query}")

        results = retriever.invoke(query)

        print(f"Retrieved {len(results)} chunks:\n")

        for i, document in enumerate(results, start=1):
            metadata = document.metadata

            print(f"--- Result {i} ---")
            print("Source:", metadata.get("source"))
            print("File type:", metadata.get("file_type"))
            print("Page:", metadata.get("page_label"))
            print("Chunk ID:", metadata.get("chunk_id"))

            print("\nContent:")
            print(document.page_content[:400])
            print()
from langchain_chroma import Chroma

from app.ingestion.embeddings import create_embedding_model


def create_vector_store():
    embeddings = create_embedding_model()

    return Chroma(
        collection_name="technova_documents",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    vector_store = create_vector_store()

    query = "What technologies does TechNova AI use?"

    results = vector_store.similarity_search_with_score(
        query,
        k=3,
    )

    print(f"Query: {query}\n")

    for i, (document, score) in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Score: {score:.4f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Chunk ID: {document.metadata.get('chunk_id')}")
        print("\nContent:")
        print(document.page_content)
        print()
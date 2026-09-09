from langchain_chroma import Chroma

from app.ingestion.embeddings import create_embedding_model


def test_metadata_filter():
    embeddings = create_embedding_model()

    vector_store = Chroma(
        collection_name="technova_documents",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )

    query = "What is the refund policy?"

    # --------------------------------
    # 1. Search all documents
    # --------------------------------

    print("\n=== SEARCH ALL DOCUMENTS ===")

    all_results = vector_store.similarity_search(
        query,
        k=5,
    )

    for index, document in enumerate(all_results, start=1):
        print(
            index,
            document.metadata.get("source"),
            document.metadata.get("file_type"),
            document.metadata.get("chunk_id"),
        )

    # --------------------------------
    # 2. Search PDF files only
    # --------------------------------

    print("\n=== SEARCH PDF ONLY ===")

    pdf_results = vector_store.similarity_search(
        query,
        k=5,
        filter={
            "file_type": "pdf"
        },
    )

    for index, document in enumerate(pdf_results, start=1):
        print(
            index,
            document.metadata.get("source"),
            document.metadata.get("file_type"),
            document.metadata.get("chunk_id"),
            document.metadata.get("page_label"),
        )

    # --------------------------------
    # 3. Search one specific document
    # --------------------------------

    print("\n=== SEARCH SPECIFIC SOURCE ===")

    source_results = vector_store.similarity_search(
        query,
        k=5,
        filter={
            "source": "refund_policy.pdf"
        },
    )

    for index, document in enumerate(source_results, start=1):
        print(
            index,
            document.metadata.get("source"),
            document.metadata.get("file_type"),
            document.metadata.get("chunk_id"),
            document.metadata.get("page_label"),
        )


if __name__ == "__main__":
    test_metadata_filter()
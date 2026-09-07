from app.retrieval.retriever import create_retriever


queries = [
    "Where is TechNova AI headquartered?",
    "What products does TechNova AI offer?",
    "What are the customer support hours?",
    "What is the refund policy?",
    "What technologies does TechNova AI use?",
]


retriever = create_retriever()

for query in queries:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = retriever.invoke(query)

   
    for i, document in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Chunk ID: {document.metadata.get('chunk_id')}")
        print(f"File type: {document.metadata.get('file_type')}")
        print("\nContent:")
        print(document.page_content)
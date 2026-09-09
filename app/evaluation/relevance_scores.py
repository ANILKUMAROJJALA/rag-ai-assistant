from app.retrieval.retriever import retrieve_documents
from app.retrieval.reranker import (
    create_reranker,
    rerank_documents,
)


reranker = create_reranker()


RELEVANT_QUESTIONS = [
    "What is TechNova AI's refund policy?",
    "Where is TechNova AI headquartered?",
    "What does NovaSearch do?",
    "How does TechNova AI protect customer passwords?",
    "What technologies does TechNova AI use?",
]


IRRELEVANT_QUESTIONS = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "How do I cook pasta?",
    "What is photosynthesis?",
    "Who won the FIFA World Cup?",
]


def evaluate_question(question):

    documents = retrieve_documents(
        question,
        metadata_filter=None,
        k=5,
    )

    ranked_documents = rerank_documents(
        question,
        documents,
        reranker,
    )

    if not ranked_documents:
        return None

    top_document, top_score = ranked_documents[0]

    return {
        "question": question,
        "score": float(top_score),
        "source": top_document.metadata.get("source"),
        "chunk_id": top_document.metadata.get("chunk_id"),
    }


def run_evaluation():

    print("\n" + "=" * 70)
    print("RELEVANT QUESTIONS")
    print("=" * 70)

    for question in RELEVANT_QUESTIONS:

        result = evaluate_question(question)

        print("\nQuestion:")
        print(result["question"])

        print("Top Score:")
        print(result["score"])

        print("Top Source:")
        print(
            result["source"],
            "chunk",
            result["chunk_id"],
        )


    print("\n" + "=" * 70)
    print("IRRELEVANT QUESTIONS")
    print("=" * 70)

    for question in IRRELEVANT_QUESTIONS:

        result = evaluate_question(question)

        print("\nQuestion:")
        print(result["question"])

        print("Top Score:")
        print(result["score"])

        print("Top Source:")
        print(
            result["source"],
            "chunk",
            result["chunk_id"],
        )


if __name__ == "__main__":
    run_evaluation()
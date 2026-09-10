from fastapi.testclient import TestClient

from app.api.main import app

import uuid

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_check():
    """
    The health endpoint should confirm
    that the API service is running.
    """

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["service"] == "RAG AI Assistant"
    
def test_empty_question_validation():
    """
    An empty question should be rejected
    by Pydantic before reaching LangGraph.
    """

    response = client.post(
        "/chat",
        json={
            "question": "",
            "thread_id": "validation-test",
        },
    )

    assert response.status_code == 422
    
def test_rag_novasearch_question():
    """
    A known document question should run through
    the RAG pipeline and return grounded sources.
    """

    thread_id = (
        f"pytest-rag-{uuid.uuid4()}"
    )

    response = client.post(
        "/chat",
        json={
            "question":
                "What does NovaSearch do?",
            "thread_id":
                thread_id,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["route"] == "rag"

    assert (
        data["retrieval_relevant"]
        is True
    )

    assert (
        data["standalone_question"]
        == "What does NovaSearch do?"
    )

    assert data["answer"]

    assert len(
        data["sources"]
    ) > 0

    source_names = {
        source["source"]
        for source
        in data["sources"]
    }

    assert (
        "product_guide.docx"
        in source_names
        or
        "company_info.txt"
        in source_names
    )
def test_conversation_memory_and_query_rewriting():
    """
    A follow-up question should use the same
    conversation thread and resolve pronouns
    using LangGraph checkpointed history.
    """

    thread_id = (
        f"pytest-memory-{uuid.uuid4()}"
    )

    first_response = client.post(
        "/chat",
        json={
            "question":
                "What does NovaSearch do?",
            "thread_id":
                thread_id,
        },
    )

    assert first_response.status_code == 200

    first_data = first_response.json()

    assert first_data["route"] == "rag"

    assert (
        first_data["retrieval_relevant"]
        is True
    )

    second_response = client.post(
        "/chat",
        json={
            "question":
                "What does it help users do?",
            "thread_id":
                thread_id,
        },
    )

    assert second_response.status_code == 200

    second_data = second_response.json()

    assert second_data["route"] == "rag"

    assert (
        second_data["retrieval_relevant"]
        is True
    )

    assert second_data["answer"]

    assert len(
        second_data["sources"]
    ) > 0

    standalone_question = (
        second_data["standalone_question"]
        or ""
    )

    assert (
        "NovaSearch"
        in standalone_question
    )
def test_rag_direct_rag_state_transition():
    """
    A direct-response turn must not inherit stale
    RAG-only state from the previous conversation turn.
    """

    thread_id = (
        f"pytest-state-{uuid.uuid4()}"
    )

    # -----------------------------
    # Turn 1: RAG
    # -----------------------------

    first_response = client.post(
        "/chat",
        json={
            "question":
                "What does NovaSearch do?",
            "thread_id":
                thread_id,
        },
    )

    assert first_response.status_code == 200

    first_data = first_response.json()

    assert first_data["route"] == "rag"

    assert (
        first_data["retrieval_relevant"]
        is True
    )

    assert len(
        first_data["sources"]
    ) > 0

    # -----------------------------
    # Turn 2: Direct response
    # -----------------------------

    second_response = client.post(
        "/chat",
        json={
            "question": "Thanks!",
            "thread_id": thread_id,
        },
    )

    assert second_response.status_code == 200

    second_data = second_response.json()

    assert second_data["route"] == "direct"

    assert (
        second_data["retrieval_relevant"]
        is None
    )

    assert (
        second_data["standalone_question"]
        == ""
    )

    assert (
        second_data["sources"]
        == []
    )

    # -----------------------------
    # Turn 3: RAG again
    # -----------------------------

    third_response = client.post(
        "/chat",
        json={
            "question":
                "What is the refund policy?",
            "thread_id":
                thread_id,
        },
    )

    assert third_response.status_code == 200

    third_data = third_response.json()

    assert third_data["route"] == "rag"

    assert (
        third_data["retrieval_relevant"]
        is True
    )

    assert third_data["answer"]

    assert len(
        third_data["sources"]
    ) > 0

    source_names = {
        source["source"]
        for source
        in third_data["sources"]
    }

    assert (
        "refund_policy.pdf"
        in source_names
        or
        "company_info.txt"
        in source_names
    )
    
def test_no_answer_for_unsupported_question():
    """
    A question that is not supported by the private
    document collection should be rejected by the
    relevance guard instead of being hallucinated.
    """

    thread_id = (
        f"pytest-no-answer-{uuid.uuid4()}"
    )

    response = client.post(
        "/chat",
        json={
            "question":
                "What is the capital of France?",
            "thread_id":
                thread_id,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["route"] == "rag"

    assert (
        data["retrieval_relevant"]
        is False
    )

    assert (
        data["sources"]
        == []
    )

    assert (
        data["answer"]
        == (
            "I don't have enough information "
            "in the provided documents to "
            "answer that."
        )
    )
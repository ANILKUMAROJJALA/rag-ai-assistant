import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.main import (
    app,
    get_rag_app,
)


client = TestClient(app)


# --------------------------------------------------
# Fake snapshot
# --------------------------------------------------

class FakeSnapshot:
    """
    Minimal object that imitates the
    LangGraph state snapshot.
    """

    def __init__(self, values=None):
        self.values = values or {}


# --------------------------------------------------
# Fake RAG application
# --------------------------------------------------

class FakeRAGApp:
    """
    Deterministic replacement for the
    real LangGraph RAG application.
    """

    def get_state(
        self,
        config,
    ):
        """
        Pretend that no previous conversation
        state exists.
        """

        return FakeSnapshot(
            values={}
        )

    def invoke(
        self,
        graph_input,
        config,
    ):
        """
        Return a fixed RAG result without
        calling the real retrieval pipeline
        or LLM.
        """

        return {
            "answer":
                "Mocked NovaSearch answer.",

            "route":
                "rag",

            "retrieval_relevant":
                True,

            "standalone_question":
                "What does NovaSearch do?",

            "sources":
                [
                    {
                        "source":
                            "product_guide.docx",

                        "file_type":
                            "docx",

                        "chunk_id":
                            "mock-chunk-1",

                        "page":
                            None,
                    }
                ],
        }


# --------------------------------------------------
# Fast API tests
# --------------------------------------------------

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

    assert (
        data["service"]
        == "RAG AI Assistant"
    )


def test_empty_question_validation():
    """
    An empty question should be rejected
    by Pydantic before reaching LangGraph.
    """

    response = client.post(
        "/chat",
        json={
            "question": "",
            "thread_id":
                "validation-test",
        },
    )

    assert response.status_code == 422


# --------------------------------------------------
# Mocked chat API test
# --------------------------------------------------

def test_chat_with_mocked_rag():
    """
    The /chat endpoint should correctly
    process and serialize a mocked
    LangGraph response.
    """

    fake_rag_app = FakeRAGApp()

    app.dependency_overrides[
        get_rag_app
    ] = lambda: fake_rag_app

    try:

        response = client.post(
            "/chat",
            json={
                "question":
                    "What does NovaSearch do?",

                "thread_id":
                    "mock-test-thread",
            },
        )

        assert (
            response.status_code
            == 200
        )

        data = response.json()

        assert (
            data["question"]
            == "What does NovaSearch do?"
        )

        assert (
            data["answer"]
            == "Mocked NovaSearch answer."
        )

        assert (
            data["thread_id"]
            == "mock-test-thread"
        )

        assert (
            data["route"]
            == "rag"
        )

        assert (
            data["retrieval_relevant"]
            is True
        )

        assert (
            data["standalone_question"]
            == "What does NovaSearch do?"
        )

        assert (
            len(data["sources"])
            == 1
        )

        assert (
            data["sources"][0]["source"]
            == "product_guide.docx"
        )

    finally:

        app.dependency_overrides.clear()


# --------------------------------------------------
# Real integration tests
# --------------------------------------------------

@pytest.mark.integration
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

    assert (
        len(data["sources"])
        > 0
    )

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


@pytest.mark.integration
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

    assert (
        first_response.status_code
        == 200
    )

    first_data = (
        first_response.json()
    )

    assert (
        first_data["route"]
        == "rag"
    )

    assert (
        first_data[
            "retrieval_relevant"
        ]
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

    assert (
        second_response.status_code
        == 200
    )

    second_data = (
        second_response.json()
    )

    assert (
        second_data["route"]
        == "rag"
    )

    assert (
        second_data[
            "retrieval_relevant"
        ]
        is True
    )

    assert second_data["answer"]

    assert (
        len(
            second_data[
                "sources"
            ]
        )
        > 0
    )

    standalone_question = (
        second_data[
            "standalone_question"
        ]
        or ""
    )

    assert (
        "NovaSearch"
        in standalone_question
    )


@pytest.mark.integration
def test_rag_direct_rag_state_transition():
    """
    A direct-response turn must not inherit stale
    RAG-only state from the previous conversation turn.
    """

    thread_id = (
        f"pytest-state-{uuid.uuid4()}"
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

    assert (
        first_response.status_code
        == 200
    )

    first_data = (
        first_response.json()
    )

    assert (
        first_data["route"]
        == "rag"
    )

    assert (
        first_data[
            "retrieval_relevant"
        ]
        is True
    )

    assert (
        len(
            first_data[
                "sources"
            ]
        )
        > 0
    )

    second_response = client.post(
        "/chat",
        json={
            "question":
                "Thanks!",

            "thread_id":
                thread_id,
        },
    )

    assert (
        second_response.status_code
        == 200
    )

    second_data = (
        second_response.json()
    )

    assert (
        second_data["route"]
        == "direct"
    )

    assert (
        second_data[
            "retrieval_relevant"
        ]
        is None
    )

    assert (
        second_data[
            "standalone_question"
        ]
        == ""
    )

    assert (
        second_data["sources"]
        == []
    )

    third_response = client.post(
        "/chat",
        json={
            "question":
                "What is the refund policy?",

            "thread_id":
                thread_id,
        },
    )

    assert (
        third_response.status_code
        == 200
    )

    third_data = (
        third_response.json()
    )

    assert (
        third_data["route"]
        == "rag"
    )

    assert (
        third_data[
            "retrieval_relevant"
        ]
        is True
    )

    assert third_data["answer"]

    assert (
        len(
            third_data[
                "sources"
            ]
        )
        > 0
    )

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


@pytest.mark.integration
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
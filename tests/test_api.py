import uuid

import pytest

from fastapi.testclient import TestClient

import app.api.documents as documents_module
import app.api.history as history_module

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

    def __init__(
        self,
        values=None,
    ):
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
        Pretend that no previous
        LangGraph state exists.
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
# Isolation fixtures
# --------------------------------------------------


@pytest.fixture(
    autouse=True
)
def isolated_history_database(
    tmp_path,
    monkeypatch,
):
    """
    Every test gets its own temporary
    conversation-history SQLite database.

    The real conversation_history.sqlite
    is never modified by these tests.
    """

    test_database = (
        tmp_path
        / "conversation_history.sqlite"
    )

    monkeypatch.setattr(
        history_module,
        "DATABASE_PATH",
        test_database,
    )

    history_module.initialize_history_database()

    yield test_database


@pytest.fixture(
    autouse=True
)
def isolated_document_directory(
    tmp_path,
    monkeypatch,
):
    """
    Redirect document filesystem
    operations to a temporary directory.

    The real data/raw directory is not
    modified by these unit tests.
    """

    raw_directory = (
        tmp_path
        / "raw"
    )

    raw_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    monkeypatch.setattr(
        documents_module,
        "RAW_DATA_DIRECTORY",
        str(raw_directory),
    )

    yield raw_directory


@pytest.fixture
def mocked_rag_app():
    """
    Override the real RAG dependency
    with a deterministic fake.
    """

    fake_rag_app = FakeRAGApp()

    app.dependency_overrides[
        get_rag_app
    ] = lambda: fake_rag_app

    yield fake_rag_app

    app.dependency_overrides.clear()


# --------------------------------------------------
# Basic FastAPI tests
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

    assert (
        data["status"]
        == "healthy"
    )

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
# Conversation API tests
# --------------------------------------------------


def test_create_conversation():
    """
    POST /conversations should create
    a new persisted conversation.
    """

    response = client.post(
        "/conversations"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"]

    assert (
        data["title"]
        == "New chat"
    )

    assert data["created_at"]
    assert data["updated_at"]


def test_list_conversations():
    """
    GET /conversations should return
    conversations that were created.
    """

    first_response = client.post(
        "/conversations"
    )

    second_response = client.post(
        "/conversations"
    )

    assert (
        first_response.status_code
        == 200
    )

    assert (
        second_response.status_code
        == 200
    )

    response = client.get(
        "/conversations"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    returned_ids = {
        item["id"]
        for item in data
    }

    assert (
        first_response.json()["id"]
        in returned_ids
    )

    assert (
        second_response.json()["id"]
        in returned_ids
    )


def test_read_conversation():
    """
    GET /conversations/{id} should
    return the selected conversation.
    """

    create_response = client.post(
        "/conversations"
    )

    assert (
        create_response.status_code
        == 200
    )

    conversation_id = (
        create_response.json()["id"]
    )

    response = client.get(
        (
            "/conversations/"
            f"{conversation_id}"
        )
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["id"]
        == conversation_id
    )

    assert (
        data["title"]
        == "New chat"
    )

    assert (
        data["messages"]
        == []
    )


def test_missing_conversation_returns_404():
    """
    Reading an unknown conversation
    should return HTTP 404.
    """

    response = client.get(
        "/conversations/does-not-exist"
    )

    assert response.status_code == 404

    assert (
        response.json()["detail"]
        == "Conversation not found."
    )


def test_delete_conversation():
    """
    DELETE /conversations/{id} should
    remove the selected conversation.
    """

    create_response = client.post(
        "/conversations"
    )

    assert (
        create_response.status_code
        == 200
    )

    conversation_id = (
        create_response.json()["id"]
    )

    delete_response = client.delete(
        (
            "/conversations/"
            f"{conversation_id}"
        )
    )

    assert (
        delete_response.status_code
        == 200
    )

    delete_data = (
        delete_response.json()
    )

    assert (
        delete_data["deleted"]
        is True
    )

    assert (
        delete_data["conversation_id"]
        == conversation_id
    )

    read_response = client.get(
        (
            "/conversations/"
            f"{conversation_id}"
        )
    )

    assert (
        read_response.status_code
        == 404
    )


def test_delete_missing_conversation_returns_404():
    """
    Deleting an unknown conversation
    should return HTTP 404.
    """

    response = client.delete(
        "/conversations/does-not-exist"
    )

    assert response.status_code == 404


# --------------------------------------------------
# Mocked chat API tests
# --------------------------------------------------


def test_chat_with_mocked_rag(
    mocked_rag_app,
):
    """
    The /chat endpoint should correctly
    process and serialize a mocked
    LangGraph response.
    """

    thread_id = (
        "mock-test-thread"
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
        == thread_id
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


def test_chat_auto_creates_conversation(
    mocked_rag_app,
):
    """
    /chat should automatically create
    application-level conversation history
    when the supplied thread ID does not
    already exist.
    """

    thread_id = (
        f"auto-create-{uuid.uuid4()}"
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

    history_response = client.get(
        (
            "/conversations/"
            f"{thread_id}"
        )
    )

    assert (
        history_response.status_code
        == 200
    )

    history = (
        history_response.json()
    )

    assert (
        history["id"]
        == thread_id
    )

    assert (
        history["title"]
        == "What does NovaSearch do?"
    )

    assert (
        len(history["messages"])
        == 2
    )

    user_message = (
        history["messages"][0]
    )

    assistant_message = (
        history["messages"][1]
    )

    assert (
        user_message["role"]
        == "user"
    )

    assert (
        user_message["content"]
        == "What does NovaSearch do?"
    )

    assert (
        assistant_message["role"]
        == "assistant"
    )

    assert (
        assistant_message["content"]
        == "Mocked NovaSearch answer."
    )

    assert (
        assistant_message["route"]
        == "rag"
    )

    assert (
        assistant_message[
            "retrieval_relevant"
        ]
        is True
    )

    assert (
        len(
            assistant_message[
                "sources"
            ]
        )
        == 1
    )

    assert (
        assistant_message[
            "sources"
        ][0]["source"]
        == "product_guide.docx"
    )


def test_chat_uses_existing_conversation(
    mocked_rag_app,
):
    """
    A frontend-created conversation
    should be reused rather than
    duplicated when /chat is called.
    """

    create_response = client.post(
        "/conversations"
    )

    assert (
        create_response.status_code
        == 200
    )

    conversation_id = (
        create_response.json()["id"]
    )

    chat_response = client.post(
        "/chat",
        json={
            "question":
                "What does NovaSearch do?",

            "thread_id":
                conversation_id,
        },
    )

    assert (
        chat_response.status_code
        == 200
    )

    conversations_response = client.get(
        "/conversations"
    )

    assert (
        conversations_response.status_code
        == 200
    )

    conversations = (
        conversations_response.json()
    )

    matching = [
        conversation
        for conversation
        in conversations
        if (
            conversation["id"]
            == conversation_id
        )
    ]

    assert len(matching) == 1


# --------------------------------------------------
# Document API tests
# --------------------------------------------------


def test_list_documents(
    isolated_document_directory,
):
    """
    GET /documents should return only
    supported document types.
    """

    text_file = (
        isolated_document_directory
        / "knowledge.txt"
    )

    text_file.write_text(
        (
            "NovaSearch provides "
            "enterprise search."
        ),
        encoding="utf-8",
    )

    ignored_file = (
        isolated_document_directory
        / "notes.md"
    )

    ignored_file.write_text(
        "This file should not appear.",
        encoding="utf-8",
    )

    response = client.get(
        "/documents"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1

    assert (
        data[0]["name"]
        == "knowledge.txt"
    )

    assert (
        data[0]["file_type"]
        == "txt"
    )

    assert (
        data[0]["status"]
        == "ready"
    )

    assert (
        data[0]["size"]
        == text_file.stat().st_size
    )


def test_reject_unsupported_document_type():
    """
    Uploading an unsupported extension
    should return HTTP 400.
    """

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.md",
                b"Unsupported file",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 400

    assert (
        "Unsupported file type"
        in response.json()["detail"]
    )


def test_reject_empty_document():
    """
    An empty supported document should
    be rejected before indexing.
    """

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == (
            "The uploaded file "
            "is empty."
        )
    )


def test_successful_txt_upload(
    isolated_document_directory,
    monkeypatch,
):
    """
    A supported TXT upload should be
    saved and indexed.

    Actual vector indexing is mocked.
    """

    indexed_paths = []

    def fake_index_file(
        file_path,
    ):
        indexed_paths.append(
            file_path
        )

        return {
            "source":
                file_path.name,

            "chunks":
                3,
        }

    monkeypatch.setattr(
        documents_module,
        "index_file",
        fake_index_file,
    )

    content = (
        b"Portfolio testing document "
        b"for NovaSearch."
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "pytest_document.txt",
                content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["name"]
        == "pytest_document.txt"
    )

    assert (
        data["file_type"]
        == "txt"
    )

    assert (
        data["status"]
        == "ready"
    )

    assert (
        data["size"]
        == len(content)
    )

    assert (
        data["chunks"]
        == 3
    )

    saved_file = (
        isolated_document_directory
        / "pytest_document.txt"
    )

    assert saved_file.exists()

    assert (
        saved_file.read_bytes()
        == content
    )

    assert len(indexed_paths) == 1

    assert (
        indexed_paths[0]
        == saved_file
    )


def test_duplicate_filename_is_rejected(
    isolated_document_directory,
    monkeypatch,
):
    """
    Uploading another document with the
    same filename should return HTTP 409
    rather than overwrite it.
    """

    existing_file = (
        isolated_document_directory
        / "duplicate.txt"
    )

    existing_file.write_text(
        "Existing document.",
        encoding="utf-8",
    )

    def should_not_index(
        file_path,
    ):
        raise AssertionError(
            (
                "index_file should not "
                "run for duplicates."
            )
        )

    monkeypatch.setattr(
        documents_module,
        "index_file",
        should_not_index,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "duplicate.txt",
                b"Replacement content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 409

    assert (
        response.json()["detail"]
        == (
            "A document with this "
            "filename already exists."
        )
    )

    assert (
        existing_file.read_text(
            encoding="utf-8"
        )
        == "Existing document."
    )


def test_filename_is_sanitized_on_upload(
    isolated_document_directory,
    monkeypatch,
):
    """
    Path components should not allow
    an upload to escape the configured
    raw-data directory.
    """

    def fake_index_file(
        file_path,
    ):
        return {
            "source":
                file_path.name,

            "chunks":
                1,
        }

    monkeypatch.setattr(
        documents_module,
        "index_file",
        fake_index_file,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "../../safe.txt",
                b"Safe content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["name"]
        == "safe.txt"
    )

    assert (
        (
            isolated_document_directory
            / "safe.txt"
        ).exists()
    )


def test_indexing_failure_removes_uploaded_file(
    isolated_document_directory,
    monkeypatch,
):
    """
    If indexing fails after the file
    is written, the upload should be
    rolled back.
    """

    def failing_index_file(
        file_path,
    ):
        raise RuntimeError(
            "Simulated indexing failure"
        )

    monkeypatch.setattr(
        documents_module,
        "index_file",
        failing_index_file,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "broken.txt",
                b"Broken indexing test",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500

    assert (
        response.json()["detail"]
        == (
            "The document was uploaded "
            "but could not be indexed."
        )
    )

    assert not (
        isolated_document_directory
        / "broken.txt"
    ).exists()


def test_delete_document(
    isolated_document_directory,
    monkeypatch,
):
    """
    DELETE /documents/{filename}
    should remove both the source
    file and its indexed chunks.
    """

    document = (
        isolated_document_directory
        / "delete_me.txt"
    )

    document.write_text(
        "Temporary test document.",
        encoding="utf-8",
    )

    deleted_sources = []

    def fake_delete_source(
        source_name,
    ):
        deleted_sources.append(
            source_name
        )

        return 2

    monkeypatch.setattr(
        documents_module,
        "delete_source_from_chroma",
        fake_delete_source,
    )

    response = client.delete(
        "/documents/delete_me.txt"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["deleted"]
        is True
    )

    assert (
        data["name"]
        == "delete_me.txt"
    )

    assert not document.exists()

    assert (
        deleted_sources
        == [
            "delete_me.txt"
        ]
    )


def test_delete_missing_document_returns_404(
    monkeypatch,
):
    """
    Deleting an unknown document
    should return HTTP 404.
    """

    def should_not_delete(
        source_name,
    ):
        raise AssertionError(
            (
                "Chroma deletion should "
                "not run for a missing file."
            )
        )

    monkeypatch.setattr(
        documents_module,
        "delete_source_from_chroma",
        should_not_delete,
    )

    response = client.delete(
        "/documents/not-there.txt"
    )

    assert response.status_code == 404

    assert (
        response.json()["detail"]
        == "Document not found."
    )


# --------------------------------------------------
# Real integration tests
# --------------------------------------------------


@pytest.mark.integration
def test_rag_novasearch_question():
    """
    A known document question should run
    through the real RAG pipeline and
    return grounded sources.
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
    A follow-up question should use the
    same conversation thread and resolve
    pronouns using LangGraph history.
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
    A direct-response turn must not
    inherit stale RAG-only state from
    the previous conversation turn.
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
            first_data["sources"]
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
            third_data["sources"]
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
    A question unsupported by the private
    document collection should be rejected
    by the relevance guard instead of
    being hallucinated.
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

    assert (
        data["route"]
        == "rag"
    )

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
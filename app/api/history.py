import json
import sqlite3
import uuid

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DATABASE_PATH = Path(
    "conversation_history.sqlite"
)


def utc_now():
    """
    Return the current UTC time
    as an ISO-formatted string.
    """
    return datetime.now(
        timezone.utc
    ).isoformat()


@contextmanager
def get_connection():
    """
    Open a SQLite connection and
    automatically commit/close it.
    """
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = (
        sqlite3.Row
    )

    # SQLite does not enforce foreign keys
    # unless this is enabled per connection.
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    try:
        yield connection
        connection.commit()

    finally:
        connection.close()


def initialize_history_database():
    """
    Create the conversation-history tables
    if they do not already exist.
    """
    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                route TEXT,
                retrieval_relevant INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (
                    conversation_id
                )
                REFERENCES conversations(id)
                ON DELETE CASCADE
            )
            """
        )


def create_conversation(
    conversation_id=None,
):
    """
    Create a conversation.

    If no ID is supplied, generate
    a UUID for frontend New Chat.

    If an ID is supplied, preserve
    that exact ID. This keeps /chat
    backward compatible with existing
    API clients and tests.
    """
    if conversation_id is None:
        conversation_id = str(
            uuid.uuid4()
        )

    now = utc_now()

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO conversations (
                id,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                "New chat",
                now,
                now,
            ),
        )

    return get_conversation(
        conversation_id
    )


def ensure_conversation(
    conversation_id: str,
):
    """
    Ensure a conversation exists.

    This is useful for /chat so older
    clients can send a thread_id directly
    without first calling POST /conversations.
    """
    existing_conversation = (
        get_conversation(
            conversation_id
        )
    )

    if existing_conversation is not None:
        return existing_conversation

    return create_conversation(
        conversation_id=conversation_id
    )


def get_conversations():
    """
    Return all conversations ordered
    by most recently updated first.
    """
    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                id,
                title,
                created_at,
                updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def get_conversation(
    conversation_id: str,
):
    """
    Return one conversation together
    with all of its persisted messages.
    """
    with get_connection() as connection:

        conversation_row = (
            connection.execute(
                """
                SELECT
                    id,
                    title,
                    created_at,
                    updated_at
                FROM conversations
                WHERE id = ?
                """,
                (
                    conversation_id,
                ),
            ).fetchone()
        )

        if conversation_row is None:
            return None

        message_rows = (
            connection.execute(
                """
                SELECT
                    id,
                    role,
                    content,
                    sources_json,
                    route,
                    retrieval_relevant,
                    created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (
                    conversation_id,
                ),
            ).fetchall()
        )

    messages = []

    for row in message_rows:

        item = dict(row)

        try:
            item["sources"] = (
                json.loads(
                    item.pop(
                        "sources_json"
                    )
                )
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            item["sources"] = []

        if (
            item[
                "retrieval_relevant"
            ]
            is not None
        ):
            item[
                "retrieval_relevant"
            ] = bool(
                item[
                    "retrieval_relevant"
                ]
            )

        messages.append(
            item
        )

    return {
        **dict(
            conversation_row
        ),
        "messages": messages,
    }


def conversation_exists(
    conversation_id: str,
):
    """
    Check whether a conversation
    already exists.
    """
    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT 1
            FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        ).fetchone()

    return row is not None


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    sources=None,
    route=None,
    retrieval_relevant=None,
):
    """
    Persist one chat message and update
    the parent conversation timestamp.
    """
    message_id = str(
        uuid.uuid4()
    )

    now = utc_now()

    sources = (
        sources
        if sources is not None
        else []
    )

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO messages (
                id,
                conversation_id,
                role,
                content,
                sources_json,
                route,
                retrieval_relevant,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                json.dumps(
                    sources
                ),
                route,
                (
                    None
                    if (
                        retrieval_relevant
                        is None
                    )
                    else int(
                        retrieval_relevant
                    )
                ),
                now,
            ),
        )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                now,
                conversation_id,
            ),
        )

    return message_id


def maybe_create_title(
    conversation_id: str,
    first_question: str,
):
    """
    Replace the default 'New chat'
    title with a short title based on
    the first user question.
    """
    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT title
            FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        ).fetchone()

        if row is None:
            return

        if (
            row["title"]
            != "New chat"
        ):
            return

        clean_question = (
            " ".join(
                first_question.split()
            )
        )

        if len(
            clean_question
        ) > 46:
            clean_question = (
                clean_question[:43]
                + "..."
            )

        title = (
            clean_question
            or "New chat"
        )

        connection.execute(
            """
            UPDATE conversations
            SET
                title = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                utc_now(),
                conversation_id,
            ),
        )


def delete_conversation(
    conversation_id: str,
):
    """
    Delete a conversation.

    Because foreign-key enforcement is
    enabled, associated messages are
    deleted through ON DELETE CASCADE.
    """
    with get_connection() as connection:

        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        )

    return cursor.rowcount > 0
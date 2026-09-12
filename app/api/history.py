import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import (
    CONVERSATION_DATABASE,
)


# ---------------------------------------------------------
# Database configuration
# ---------------------------------------------------------

# Default database path comes from config.py.
#
# DATABASE_PATH remains module-level intentionally because
# our pytest fixtures monkeypatch it to a temporary SQLite
# database for test isolation.
DATABASE_PATH = Path(
    CONVERSATION_DATABASE
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def utc_now():
    """
    Return the current UTC timestamp
    in ISO-8601 format.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def get_database_path():
    """
    Return the active conversation-history
    database path.

    Normally this comes from
    CONVERSATION_DATABASE.

    During automated tests, DATABASE_PATH
    can be monkeypatched to a temporary
    SQLite database.
    """

    database_path = Path(
        DATABASE_PATH
    )

    parent = database_path.parent

    if str(parent) not in (
        "",
        ".",
    ):
        parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    return database_path


def get_connection():
    """
    Create a SQLite connection to the
    conversation-history database.
    """

    connection = sqlite3.connect(
        get_database_path(),
        check_same_thread=False,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ---------------------------------------------------------
# Database initialization
# ---------------------------------------------------------


def initialize_history_database():
    """
    Create the application-level conversation
    history tables.

    This database stores the conversations
    and messages visible in the UI.

    LangGraph checkpoint state is stored
    separately.
    """

    with get_connection() as connection:

        # -------------------------------------------------
        # Conversations table
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Messages table
        # -------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sources TEXT,
                route TEXT,
                retrieval_relevant INTEGER,
                FOREIGN KEY (
                    conversation_id
                )
                REFERENCES conversations(id)
                ON DELETE CASCADE
            )
            """
        )

        # -------------------------------------------------
        # Backward-compatible schema migration
        # -------------------------------------------------

        columns = connection.execute(
            """
            PRAGMA table_info(messages)
            """
        ).fetchall()

        column_names = {
            column["name"]
            for column in columns
        }

        if "route" not in column_names:

            connection.execute(
                """
                ALTER TABLE messages
                ADD COLUMN route TEXT
                """
            )

        if (
            "retrieval_relevant"
            not in column_names
        ):

            connection.execute(
                """
                ALTER TABLE messages
                ADD COLUMN retrieval_relevant INTEGER
                """
            )

        # -------------------------------------------------
        # Indexes
        # -------------------------------------------------

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_conversation_id
            ON messages (
                conversation_id
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_conversations_updated_at
            ON conversations (
                updated_at
            )
            """
        )

        connection.commit()


# ---------------------------------------------------------
# Conversation operations
# ---------------------------------------------------------


def create_conversation(
    conversation_id=None,
):
    """
    Create a new conversation.

    If a conversation ID is supplied,
    it can also be reused as the LangGraph
    thread_id.
    """

    if conversation_id is None:

        conversation_id = str(
            uuid.uuid4()
        )

    timestamp = utc_now()

    with get_connection() as connection:

        existing = connection.execute(
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

        if existing is not None:

            return dict(
                existing
            )

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
                timestamp,
                timestamp,
            ),
        )

        connection.commit()

    return {
        "id": conversation_id,
        "title": "New chat",
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def ensure_conversation(
    conversation_id,
):
    """
    Ensure that a conversation exists.

    This keeps /chat backward compatible
    with clients that provide a new
    thread_id without first calling
    POST /conversations.
    """

    with get_connection() as connection:

        conversation = connection.execute(
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

    if conversation is not None:

        return dict(
            conversation
        )

    return create_conversation(
        conversation_id=
            conversation_id
    )


def get_conversations():
    """
    Return all conversations ordered
    by most recent activity.
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
    conversation_id,
):
    """
    Return one conversation together with
    all of its visible messages.

    Database rows are mapped into the exact
    shape expected by the FastAPI
    ConversationDetail response model.
    """

    with get_connection() as connection:

        conversation = connection.execute(
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

        if conversation is None:

            return None

        message_rows = connection.execute(
            """
            SELECT
                id,
                conversation_id,
                role,
                content,
                timestamp,
                sources,
                route,
                retrieval_relevant
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (
                conversation_id,
            ),
        ).fetchall()

    messages = []

    for row in message_rows:

        raw_message = dict(
            row
        )

        # ---------------------------------------------
        # Deserialize sources JSON
        # ---------------------------------------------

        raw_sources = raw_message.get(
            "sources"
        )

        if raw_sources:

            try:

                sources = json.loads(
                    raw_sources
                )

            except json.JSONDecodeError:

                sources = []

        else:

            sources = []

        # ---------------------------------------------
        # Convert SQLite relevance integer
        # back into Python bool / None
        # ---------------------------------------------

        retrieval_relevant = (
            raw_message.get(
                "retrieval_relevant"
            )
        )

        if retrieval_relevant is not None:

            retrieval_relevant = bool(
                retrieval_relevant
            )

        # ---------------------------------------------
        # Map database representation
        # to API response representation
        # ---------------------------------------------

        message = {
            "id": str(
                raw_message["id"]
            ),
            "role": raw_message[
                "role"
            ],
            "content": raw_message[
                "content"
            ],
            "sources": sources,
            "route": raw_message.get(
                "route"
            ),
            "retrieval_relevant":
                retrieval_relevant,
            "created_at":
                raw_message[
                    "timestamp"
                ],
        }

        messages.append(
            message
        )

    result = dict(
        conversation
    )

    result[
        "messages"
    ] = messages

    return result


def delete_conversation(
    conversation_id,
):
    """
    Delete a conversation.

    Associated visible messages are removed
    automatically because the foreign key
    uses ON DELETE CASCADE.

    LangGraph checkpoint data is stored
    separately.
    """

    with get_connection() as connection:

        existing = connection.execute(
            """
            SELECT id
            FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        ).fetchone()

        if existing is None:

            return False

        connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        )

        connection.commit()

    return True


# ---------------------------------------------------------
# Message operations
# ---------------------------------------------------------


def save_message(
    conversation_id,
    role,
    content,
    sources=None,
    route=None,
    retrieval_relevant=None,
):
    """
    Persist one visible conversation message.

    Parameters
    ----------
    conversation_id:
        Conversation ID shared with the UI
        and LangGraph thread when applicable.

    role:
        Usually "user" or "assistant".

    content:
        Visible message content.

    sources:
        RAG source metadata.

    route:
        LangGraph route used for the response,
        for example "direct" or "rag".

    retrieval_relevant:
        Relevance-guard result.

        True:
            Retrieved context was relevant.

        False:
            Retrieved context was rejected.

        None:
            Relevance checking was not
            applicable.
    """

    ensure_conversation(
        conversation_id
    )

    timestamp = utc_now()

    # -------------------------------------------------
    # Serialize source metadata
    # -------------------------------------------------

    serialized_sources = None

    if sources is not None:

        serialized_sources = (
            json.dumps(
                sources,
                ensure_ascii=False,
            )
        )

    # -------------------------------------------------
    # Convert Python boolean to SQLite integer
    # -------------------------------------------------

    serialized_relevance = None

    if retrieval_relevant is not None:

        serialized_relevance = int(
            bool(
                retrieval_relevant
            )
        )

    # -------------------------------------------------
    # Save message
    # -------------------------------------------------

    with get_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO messages (
                conversation_id,
                role,
                content,
                timestamp,
                sources,
                route,
                retrieval_relevant
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                timestamp,
                serialized_sources,
                route,
                serialized_relevance,
            ),
        )

        # Move recently-active conversations
        # to the top of the sidebar.

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                timestamp,
                conversation_id,
            ),
        )

        connection.commit()

        message_id = (
            cursor.lastrowid
        )

    # Return using API-friendly representation.
    return {
        "id": str(
            message_id
        ),
        "role": role,
        "content": content,
        "sources":
            sources or [],
        "route": route,
        "retrieval_relevant":
            retrieval_relevant,
        "created_at":
            timestamp,
    }


# ---------------------------------------------------------
# Automatic conversation titles
# ---------------------------------------------------------


def maybe_create_title(
    conversation_id,
    user_message,
):
    """
    Create a lightweight conversation title
    from the first user message.

    This avoids making an extra LLM request
    only for sidebar title generation.
    """

    if not user_message:

        return None

    cleaned_message = " ".join(
        str(
            user_message
        ).split()
    )

    if not cleaned_message:

        return None

    with get_connection() as connection:

        conversation = connection.execute(
            """
            SELECT title
            FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        ).fetchone()

        if conversation is None:

            return None

        current_title = (
            conversation[
                "title"
            ]
        )

        # Only replace placeholder titles.

        if current_title not in (
            "",
            "New chat",
            "New Chat",
            "Untitled",
        ):

            return current_title

        max_length = 46

        if len(
            cleaned_message
        ) > max_length:

            new_title = (
                cleaned_message[
                    :max_length
                ].rstrip()
                + "..."
            )

        else:

            new_title = (
                cleaned_message
            )

        timestamp = utc_now()

        connection.execute(
            """
            UPDATE conversations
            SET
                title = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                new_title,
                timestamp,
                conversation_id,
            ),
        )

        connection.commit()

    return new_title
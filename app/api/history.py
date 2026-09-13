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

# Keep this module-level intentionally.
#
# Our pytest fixtures monkeypatch DATABASE_PATH so every
# test can use its own temporary SQLite database.
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
    Create and migrate the application-level
    conversation-history database.

    This database stores conversations and
    messages visible in the UI.

    LangGraph checkpoint state is stored
    separately.

    Supported database states:
    - Fresh database
    - Current database schema
    - Legacy messages schema using:
        * TEXT message IDs
        * created_at
        * sources_json
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
        #
        # For a brand-new database this creates the
        # current schema.
        #
        # If an older messages table already exists,
        # SQLite leaves that table unchanged here.
        # We inspect and migrate it below.
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
        # Inspect the existing messages schema
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

        column_types = {
            column["name"]:
                (
                    column["type"]
                    or ""
                ).upper()
            for column in columns
        }

        # -------------------------------------------------
        # Detect legacy schema
        #
        # Legacy:
        #
        # id TEXT
        # created_at
        # sources_json
        #
        # Current:
        #
        # id INTEGER AUTOINCREMENT
        # timestamp
        # sources
        # -------------------------------------------------

        needs_table_migration = (
            column_types.get(
                "id",
                "",
            )
            != "INTEGER"
            or "timestamp"
            not in column_names
            or "sources"
            not in column_names
        )

        if needs_table_migration:

            # ---------------------------------------------
            # Resolve legacy timestamp column
            # ---------------------------------------------

            if "timestamp" in column_names:
                timestamp_column = (
                    "timestamp"
                )

            elif "created_at" in column_names:
                timestamp_column = (
                    "created_at"
                )

            else:
                raise RuntimeError(
                    "Cannot migrate messages table: "
                    "no timestamp or created_at "
                    "column was found."
                )

            # ---------------------------------------------
            # Resolve legacy sources column
            # ---------------------------------------------

            if "sources" in column_names:
                sources_column = (
                    "sources"
                )

            elif "sources_json" in column_names:
                sources_column = (
                    "sources_json"
                )

            else:
                sources_column = (
                    "NULL"
                )

            # ---------------------------------------------
            # Optional legacy columns
            # ---------------------------------------------

            if "route" in column_names:
                route_column = (
                    "route"
                )
            else:
                route_column = (
                    "NULL"
                )

            if (
                "retrieval_relevant"
                in column_names
            ):
                relevance_column = (
                    "retrieval_relevant"
                )
            else:
                relevance_column = (
                    "NULL"
                )

            # ---------------------------------------------
            # Create replacement table using
            # the current schema
            # ---------------------------------------------

            connection.execute(
                """
                DROP TABLE IF EXISTS
                messages_migrated
                """
            )

            connection.execute(
                """
                CREATE TABLE messages_migrated (
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

            # ---------------------------------------------
            # Copy old messages.
            #
            # Legacy message IDs are intentionally not
            # copied because the new schema uses integer
            # AUTOINCREMENT IDs.
            #
            # conversation_id remains unchanged, so every
            # message stays connected to the same chat.
            # ---------------------------------------------

            migration_query = f"""
                INSERT INTO messages_migrated (
                    conversation_id,
                    role,
                    content,
                    timestamp,
                    sources,
                    route,
                    retrieval_relevant
                )
                SELECT
                    conversation_id,
                    role,
                    content,
                    {timestamp_column},
                    {sources_column},
                    {route_column},
                    {relevance_column}
                FROM messages
                ORDER BY rowid ASC
            """

            connection.execute(
                migration_query
            )

            # ---------------------------------------------
            # Replace legacy table
            # ---------------------------------------------

            connection.execute(
                """
                DROP TABLE messages
                """
            )

            connection.execute(
                """
                ALTER TABLE messages_migrated
                RENAME TO messages
                """
            )

        # -------------------------------------------------
        # Re-read schema after possible migration
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

        # -------------------------------------------------
        # Simple additive migrations
        # -------------------------------------------------

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

    A UUID is generated unless a specific
    conversation ID is supplied.
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

    This preserves backward compatibility with
    clients that call POST /chat directly with a
    newly generated thread_id without first calling
    POST /conversations.
    """

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

    return create_conversation(
        conversation_id=conversation_id
    )


def get_conversations():
    """
    Return conversations ordered by most
    recently active first.
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
    Return one conversation and all of its
    visible UI messages.

    Returns None when the conversation
    does not exist.
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

        raw_messages = connection.execute(
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

    for raw_message in raw_messages:

        # -------------------------------------------------
        # Deserialize source metadata
        # -------------------------------------------------

        sources = []

        serialized_sources = (
            raw_message[
                "sources"
            ]
        )

        if serialized_sources:

            try:
                loaded_sources = json.loads(
                    serialized_sources
                )

                if isinstance(
                    loaded_sources,
                    list,
                ):
                    sources = loaded_sources

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                sources = []

        # -------------------------------------------------
        # Convert SQLite relevance representation
        # -------------------------------------------------

        raw_relevance = (
            raw_message[
                "retrieval_relevant"
            ]
        )

        if raw_relevance is None:
            retrieval_relevant = None
        else:
            retrieval_relevant = bool(
                raw_relevance
            )

        # -------------------------------------------------
        # Build API message shape
        # -------------------------------------------------

        message = {
            "id": str(
                raw_message["id"]
            ),
            "role":
                raw_message[
                    "role"
                ],
            "content":
                raw_message[
                    "content"
                ],
            "sources":
                sources,
            "route":
                raw_message[
                    "route"
                ],
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

    result["messages"] = (
        messages
    )

    return result


def delete_conversation(
    conversation_id,
):
    """
    Delete a conversation and all of its
    visible UI messages.

    SQLite ON DELETE CASCADE removes messages
    belonging to the deleted conversation.

    LangGraph checkpoint state is stored
    separately and is not modified here.
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
    Save one user or assistant message.

    Source metadata is stored as JSON.

    retrieval_relevant is stored using SQLite's
    INTEGER representation:
    True  -> 1
    False -> 0
    None  -> NULL
    """

    # -------------------------------------------------
    # Ensure parent conversation exists.
    #
    # This also preserves compatibility with clients
    # that call /chat using a brand-new thread ID.
    # -------------------------------------------------

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
    # Convert relevance value for SQLite
    # -------------------------------------------------

    if retrieval_relevant is None:
        serialized_relevance = None
    else:
        serialized_relevance = int(
            bool(
                retrieval_relevant
            )
        )

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

        # -------------------------------------------------
        # Move recently active conversations
        # to the top of the sidebar.
        # -------------------------------------------------

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

    return {
        "id": str(
            message_id
        ),
        "role": role,
        "content": content,
        "sources":
            sources
            if sources is not None
            else [],
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
    Create a lightweight title from the first
    user message.

    No additional LLM call is required.

    Existing custom/generated titles are left
    unchanged.
    """

    if user_message is None:
        return None

    cleaned_message = " ".join(
        str(
            user_message
        ).split()
    ).strip()

    if not cleaned_message:
        return None

    with get_connection() as connection:

        conversation = connection.execute(
            """
            SELECT
                id,
                title
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

        # Only automatically title a conversation
        # that still has the default title.
        if current_title != "New chat":
            return current_title

        # -------------------------------------------------
        # Create compact sidebar title
        # -------------------------------------------------

        max_length = 46

        if len(
            cleaned_message
        ) <= max_length:

            new_title = (
                cleaned_message
            )

        else:

            new_title = (
                cleaned_message[
                    :max_length
                ].rstrip()
                + "..."
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
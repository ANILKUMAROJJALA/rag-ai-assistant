import hashlib
import json
from pathlib import Path

from app.config import (
    RAW_DATA_DIRECTORY,
    INGESTION_MANIFEST_PATH,
)

from app.ingestion.loader import (
    load_document,
    SUPPORTED_EXTENSIONS,
)

from app.ingestion.chunking import (
    split_documents,
)

from app.ingestion.vectorstore import (
    create_vector_store,
)


RAW_DATA_FOLDER = Path(
    RAW_DATA_DIRECTORY
)

MANIFEST_PATH = Path(
    INGESTION_MANIFEST_PATH
)


def calculate_file_hash(
    file_path: Path,
) -> str:
    """
    Calculate the SHA-256 hash of a file.

    If the file content changes,
    the hash changes.
    """

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha256.update(
                block
            )

    return sha256.hexdigest()


def load_manifest():
    """
    Load the previous ingestion state.

    Returns an empty dictionary
    if no manifest exists yet.
    """

    if not MANIFEST_PATH.exists():
        return {}

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def save_manifest(
    manifest,
):
    """
    Save the current ingestion state.
    """

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )


def get_supported_files():
    """
    Return supported files from
    the configured raw-data directory.
    """

    if not RAW_DATA_FOLDER.exists():

        raise FileNotFoundError(
            f"Raw data directory not found: "
            f"{RAW_DATA_FOLDER}"
        )

    files = []

    for file_path in (
        RAW_DATA_FOLDER.iterdir()
    ):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        files.append(
            file_path
        )

    return sorted(
        files
    )


def delete_source_chunks(
    vector_store,
    source_name,
):
    """
    Delete every Chroma chunk belonging
    to a particular source file.
    """

    existing = vector_store.get(
        where={
            "source": source_name
        }
    )

    ids = existing.get(
        "ids",
        [],
    )

    if not ids:

        print(
            f"No existing chunks found "
            f"for: {source_name}"
        )

        return 0

    vector_store.delete(
        ids=ids
    )

    print(
        f"Deleted {len(ids)} old chunks "
        f"for: {source_name}"
    )

    return len(ids)


def ingest_file(
    vector_store,
    file_path,
):
    """
    Load, chunk, and ingest one file
    into the vector store.
    """

    documents = load_document(
        str(file_path)
    )

    chunks = split_documents(
        documents
    )

    if not chunks:
        return 0

    ids = [
        chunk.metadata[
            "chunk_id"
        ]
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return len(chunks)


def run_incremental_ingestion():
    """
    Synchronize the configured source folder
    with the Chroma vector database.

    New file:
        ingest

    Changed file:
        delete old source chunks
        ingest updated chunks

    Unchanged file:
        skip

    Deleted file:
        remove old source chunks
    """

    print(
        "\nStarting incremental ingestion..."
    )

    print(
        "Raw data directory:",
        RAW_DATA_FOLDER,
    )

    print(
        "Manifest path:",
        MANIFEST_PATH,
    )

    vector_store = (
        create_vector_store()
    )

    old_manifest = (
        load_manifest()
    )

    current_files = (
        get_supported_files()
    )

    current_manifest = {}

    current_file_names = {
        file_path.name
        for file_path
        in current_files
    }


    # --------------------------------------------------
    # Detect deleted files
    # --------------------------------------------------

    for old_source in (
        old_manifest
    ):

        if (
            old_source
            not in current_file_names
        ):

            print(
                f"\nDeleted file detected: "
                f"{old_source}"
            )

            delete_source_chunks(
                vector_store,
                old_source,
            )


    # --------------------------------------------------
    # Process files currently on disk
    # --------------------------------------------------

    for file_path in (
        current_files
    ):

        source_name = (
            file_path.name
        )

        file_hash = (
            calculate_file_hash(
                file_path
            )
        )

        current_manifest[
            source_name
        ] = {
            "hash": file_hash
        }


        # --------------------------------------------------
        # New file
        # --------------------------------------------------

        if (
            source_name
            not in old_manifest
        ):

            print(
                f"\nNew file detected: "
                f"{source_name}"
            )

            chunk_count = (
                ingest_file(
                    vector_store,
                    file_path,
                )
            )

            print(
                f"Ingested "
                f"{chunk_count} chunks."
            )

            continue


        # --------------------------------------------------
        # Existing file
        # --------------------------------------------------

        old_hash = (
            old_manifest[
                source_name
            ].get(
                "hash"
            )
        )


        # --------------------------------------------------
        # Unchanged file
        # --------------------------------------------------

        if (
            old_hash
            == file_hash
        ):

            print(
                f"\nUnchanged: "
                f"{source_name}"
            )

            print(
                "Skipping embedding."
            )

            continue


        # --------------------------------------------------
        # Changed file
        # --------------------------------------------------

        print(
            f"\nChanged file detected: "
            f"{source_name}"
        )

        delete_source_chunks(
            vector_store,
            source_name,
        )

        chunk_count = (
            ingest_file(
                vector_store,
                file_path,
            )
        )

        print(
            f"Ingested "
            f"{chunk_count} "
            f"updated chunks."
        )


    # --------------------------------------------------
    # Save new manifest
    # --------------------------------------------------

    save_manifest(
        current_manifest
    )


    # --------------------------------------------------
    # Final vector-store status
    # --------------------------------------------------

    stored = (
        vector_store.get()
    )

    print(
        "\nIncremental ingestion complete."
    )

    print(
        "Total records in Chroma:",
        len(
            stored["ids"]
        ),
    )


if __name__ == "__main__":
    run_incremental_ingestion()
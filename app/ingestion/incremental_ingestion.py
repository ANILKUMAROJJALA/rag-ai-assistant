import hashlib
import json
from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunking import split_documents
from app.ingestion.vectorstore import create_vector_store


RAW_DATA_FOLDER = Path("data/raw")
MANIFEST_PATH = Path("data/processed/ingestion_manifest.json")


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.

    If the file content changes, the hash changes.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def load_manifest():
    """
    Load the previous ingestion state.
    """

    if not MANIFEST_PATH.exists():
        return {}

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_manifest(manifest):
    """
    Save current ingestion state.
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
    Return supported files from data/raw.
    """

    supported_extensions = {
        ".txt",
        ".pdf",
        ".docx",
    }

    files = []

    for file_path in RAW_DATA_FOLDER.iterdir():

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in supported_extensions
        ):
            continue

        files.append(file_path)

    return files


def delete_source_chunks(
    vector_store,
    source_name,
):
    """
    Delete all chunks belonging to one source file.
    """

    existing = vector_store.get(
        where={
            "source": source_name
        }
    )

    ids = existing.get("ids", [])

    if ids:

        vector_store.delete(
            ids=ids
        )

        print(
            f"Deleted {len(ids)} old chunks "
            f"for: {source_name}"
        )


def ingest_file(
    vector_store,
    file_path,
):
    """
    Load, chunk, and ingest one file.
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
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return len(chunks)


def run_incremental_ingestion():
    """
    Incremental ingestion workflow.

    New file:
        ingest

    Changed file:
        delete old chunks
        ingest new chunks

    Unchanged file:
        skip

    Deleted file:
        remove old chunks from Chroma
    """

    print(
        "\nStarting incremental ingestion..."
    )

    vector_store = create_vector_store()

    old_manifest = load_manifest()

    current_files = get_supported_files()

    current_manifest = {}

    current_file_names = {
        file_path.name
        for file_path in current_files
    }


    # --------------------------------------------------
    # Detect deleted files
    # --------------------------------------------------

    for old_source in old_manifest:

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
    # Process current files
    # --------------------------------------------------

    for file_path in current_files:

        source_name = file_path.name

        file_hash = calculate_file_hash(
            file_path
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

            chunk_count = ingest_file(
                vector_store,
                file_path,
            )

            print(
                f"Ingested {chunk_count} chunks."
            )

            continue


        # --------------------------------------------------
        # Existing file
        # --------------------------------------------------

        old_hash = (
            old_manifest[
                source_name
            ].get("hash")
        )


        # --------------------------------------------------
        # Unchanged file
        # --------------------------------------------------

        if old_hash == file_hash:

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

        chunk_count = ingest_file(
            vector_store,
            file_path,
        )

        print(
            f"Ingested {chunk_count} "
            f"updated chunks."
        )


    # --------------------------------------------------
    # Save new manifest
    # --------------------------------------------------

    save_manifest(
        current_manifest
    )


    stored = vector_store.get()

    print(
        "\nIncremental ingestion complete."
    )

    print(
        "Total records in Chroma:",
        len(stored["ids"]),
    )


if __name__ == "__main__":
    run_incremental_ingestion()
import re
from pathlib import Path

from fastapi import (
    HTTPException,
    UploadFile,
)

from app.config import (
    RAW_DATA_DIRECTORY,
)

from app.ingestion.live_ingestion import (
    SUPPORTED_EXTENSIONS,
    delete_source_from_chroma,
    index_file,
)


MAX_UPLOAD_BYTES = (
    20 * 1024 * 1024
)


def get_raw_directory():
    directory = Path(
        RAW_DATA_DIRECTORY
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def safe_filename(
    filename: str,
):
    filename = Path(
        filename
    ).name

    filename = re.sub(
        r"[^A-Za-z0-9._ -]",
        "_",
        filename,
    )

    return filename.strip()


def list_documents():
    directory = (
        get_raw_directory()
    )

    documents = []

    for path in sorted(
        directory.iterdir(),
        key=lambda item:
            item.name.lower(),
    ):

        if (
            not path.is_file()
            or path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        stat = path.stat()

        documents.append(
            {
                "name": path.name,
                "file_type": (
                    path.suffix
                    .lower()
                    .replace(
                        ".",
                        "",
                    )
                ),
                "size": stat.st_size,
                "status": "ready",
            }
        )

    return documents


async def save_uploaded_file(
    upload: UploadFile,
):
    if not upload.filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file "
                "has no filename."
            ),
        )

    filename = safe_filename(
        upload.filename
    )

    extension = (
        Path(
            filename
        ).suffix.lower()
    )

    if (
        extension
        not in SUPPORTED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Upload a PDF, DOCX, "
                "or TXT file."
            ),
        )

    data = await upload.read(
        MAX_UPLOAD_BYTES + 1
    )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "File is too large. "
                "Maximum upload size "
                "is 20 MB."
            ),
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file "
                "is empty."
            ),
        )

    destination = (
        get_raw_directory()
        / filename
    )

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "A document with this "
                "filename already exists."
            ),
        )

    destination.write_bytes(
        data
    )

    try:
        result = index_file(
            destination
        )

    except Exception as error:

        destination.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The document was uploaded "
                "but could not be indexed."
            ),
        ) from error

    return {
        "name": filename,
        "file_type": (
            extension.replace(
                ".",
                "",
            )
        ),
        "size": len(data),
        "status": "ready",
        "chunks": result[
            "chunks"
        ],
    }


def delete_document(
    filename: str,
):
    filename = safe_filename(
        filename
    )

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    file_path = (
        get_raw_directory()
        / filename
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Document not found."
            ),
        )

    try:
        delete_source_from_chroma(
            filename
        )

        file_path.unlink()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "The document could "
                "not be deleted."
            ),
        ) from error

    return {
        "deleted": True,
        "name": filename,
    }
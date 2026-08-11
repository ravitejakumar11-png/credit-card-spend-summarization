from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.core.db import EmbeddingServiceError

from src.api.v1.services.upload_service import (
    ingest_document,
    clear_ingested_data,
    delete_ingested_document,
    get_ingested_documents,
)

router = APIRouter(prefix="/api/v1/upload")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}

# Safety limit for uploaded files.
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Upload document
# ---------------------------------------------------------------------------


@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF and run the ingestion pipeline.

    Responsibilities of this route:
        1. Validate the uploaded file.
        2. Check whether the same document is already present.
        3. Save the file when required.
        4. Delegate ingestion to the service layer.
        5. Convert known failures into appropriate HTTP responses.

    Duplicate handling:
        - Same filename + same byte size -> skip ingestion.
        - Same filename + different byte size -> re-ingest.
        - New filename -> ingest normally.
    """

    # -----------------------------------------------------------------------
    # Validate filename
    # -----------------------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    original_filename = Path(file.filename).name

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid file name.",
        )

    # -----------------------------------------------------------------------
    # Validate extension
    # -----------------------------------------------------------------------

    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # -----------------------------------------------------------------------
    # Read uploaded file
    # -----------------------------------------------------------------------

    try:

        contents = await file.read()

    except Exception as exc:

        print(f"[upload_route] Failed to read uploaded file: {exc}")

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded file.",
        ) from exc

    finally:

        await file.close()

    # -----------------------------------------------------------------------
    # Validate file content
    # -----------------------------------------------------------------------

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                f"File is too large. Maximum allowed size is "
                f"{MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
            ),
        )

    # -----------------------------------------------------------------------
    # Basic PDF signature validation
    # -----------------------------------------------------------------------

    if not contents.startswith(b"%PDF"):

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    # -----------------------------------------------------------------------
    # Build destination path
    #
    # Path(file.filename).name prevents filenames such as:
    #
    #     ../../some_file.pdf
    #
    # from escaping the data directory.
    # -----------------------------------------------------------------------

    file_path = DATA_DIR / original_filename

    # -----------------------------------------------------------------------
    # Check whether the same document is already present
    #
    # IMPORTANT:
    # This check MUST happen before write_bytes().
    #
    # Otherwise the existing file would already have been overwritten and
    # we would have no way to compare the incoming file against it.
    # -----------------------------------------------------------------------

    if file_path.exists():

        if not file_path.is_file():

            raise HTTPException(
                status_code=500,
                detail=(
                    "A file or directory with the same name already "
                    "exists in the data directory."
                ),
            )

        try:

            existing_file_size = file_path.stat().st_size
            uploaded_file_size = len(contents)

        except OSError as exc:

            print("[upload_route] Failed to inspect existing file: " f"{exc}")

            raise HTTPException(
                status_code=500,
                detail="Unable to inspect the existing document.",
            ) from exc

        print(f"[upload_route] Existing file found: " f"{file_path}")

        print(f"[upload_route] Existing file size : " f"{existing_file_size} bytes")

        print(f"[upload_route] Uploaded file size : " f"{uploaded_file_size} bytes")

        # ---------------------------------------------------------------
        # Same filename + same byte size
        # ---------------------------------------------------------------

        if existing_file_size == uploaded_file_size:

            print(
                "[upload_route] Same filename and byte size detected. "
                "Skipping upload and ingestion."
            )

            return {
                "status": "skipped",
                "message": (
                    "Document already exists with the same "
                    "filename and byte size. Ingestion skipped."
                ),
                "file_name": original_filename,
                "file_size": uploaded_file_size,
            }

        # ---------------------------------------------------------------
        # Same filename + different byte size
        # ---------------------------------------------------------------

        print(
            "[upload_route] Same filename but different byte size "
            "detected. Existing document will be replaced and "
            "re-ingested."
        )

    else:

        print(f"[upload_route] New document detected: " f"{original_filename}")

    # -----------------------------------------------------------------------
    # Save uploaded file
    # -----------------------------------------------------------------------

    try:

        file_path.write_bytes(contents)

        print(f"[upload_route] File uploaded: {file_path}")

    except OSError as exc:

        print(f"[upload_route] Failed to save file: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Unable to save the uploaded file.",
        ) from exc

    # -----------------------------------------------------------------------
    # Run ingestion
    # -----------------------------------------------------------------------

    try:

        result = ingest_document(str(file_path))

        return result

    except FileNotFoundError as exc:

        print(f"[upload_route] File not found during ingestion: {exc}")

        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be processed.",
        ) from exc

    except ValueError as exc:

        print(f"[upload_route] Invalid document: {exc}")

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except EmbeddingServiceError as exc:

        print("[upload_route] Embedding service unavailable: " f"{exc}")

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print("[upload_route] Ingestion failed: " f"{exc}")

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from exc


# ---------------------------------------------------------------------------
# Clear all ingested data
# ---------------------------------------------------------------------------


@router.delete("/clear")
def clear_all_ingested_data():
    """
    Delete all ingested documents, chunks, source files and extracted images.

    This is intentionally separate from normal upload.

    Normal upload:
        ADD / RE-INGEST a document.

    DELETE /clear:
        Completely reset the ingestion environment.
    """

    try:

        result = clear_ingested_data()

        return {
            "status": "success",
            "message": "All ingested data has been cleared.",
            **result,
        }

    except Exception as exc:

        print(f"[upload_route] Failed to clear ingested data: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to clear ingested data. "
                "Please check the server logs for details."
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Delete one document
# ---------------------------------------------------------------------------


@router.delete("/{doc_id}")
def delete_document_endpoint(doc_id: str):
    """
    Delete one ingested document and all associated data.
    """

    try:

        result = delete_ingested_document(doc_id)

        return {
            "status": "success",
            "message": "Document and associated data deleted.",
            **result,
        }

    except ValueError as exc:

        print(f"[upload_route] Document not found: {exc}")

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(f"[upload_route] Failed to delete document: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Failed to delete document.",
        ) from exc


# ---------------------------------------------------------------------------
# List ingested documents
# ---------------------------------------------------------------------------


@router.get("/")
def list_documents():
    """
    Return all currently ingested documents.
    """

    try:

        documents = get_ingested_documents()

        return {
            "status": "success",
            "documents": documents,
        }

    except Exception as exc:

        print(f"[upload_route] Failed to list documents: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve ingested documents.",
        ) from exc

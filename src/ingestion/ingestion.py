import pathlib

from dotenv import load_dotenv

from src.core.db import (
    delete_all_ingested_data,
    delete_document_by_id,
    store_chunks,
    upsert_document,
)
from src.ingestion.docling_parser import parse_document

load_dotenv()


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IngestionError(Exception):
    """
    Base exception for ingestion failures.
    """

    pass


class DocumentValidationError(IngestionError):
    """
    Raised when the source document is invalid or cannot be processed.
    """

    pass


class DocumentParsingError(IngestionError):
    """
    Raised when Docling cannot parse the document.
    """

    pass


class DocumentStorageError(IngestionError):
    """
    Raised when chunks cannot be embedded/stored.
    """

    pass


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split a long string into overlapping character windows.

    Splitting strategy:
        - Each chunk has a maximum size of chunk_size characters.
        - Adjacent chunks overlap by overlap characters.
        - The overlap preserves context across chunk boundaries.

    Tables and images are not passed through this function.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[str] = []

    step = chunk_size - overlap
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

        start += step

    return chunks


# ---------------------------------------------------------------------------
# Chunk preparation
# ---------------------------------------------------------------------------


def _prepare_chunks(parsed_elements: list[dict]) -> list[dict]:
    """
    Split long text elements while keeping tables and images atomic.
    """

    chunks: list[dict] = []

    for index, element in enumerate(parsed_elements):

        if not isinstance(element, dict):
            raise DocumentParsingError(
                f"Parser returned an invalid element at index {index}."
            )

        if "content" not in element:
            raise DocumentParsingError(
                f"Parser element at index {index} is missing 'content'."
            )

        if "content_type" not in element:
            raise DocumentParsingError(
                f"Parser element at index {index} is missing " "'content_type'."
            )

        if "metadata" not in element:
            raise DocumentParsingError(
                f"Parser element at index {index} is missing 'metadata'."
            )

        content = element["content"]
        content_type = element["content_type"]

        if not isinstance(content, str):
            raise DocumentParsingError(
                f"Parser element at index {index} has invalid content."
            )

        if not content.strip():
            continue

        if content_type == "text" and len(content) > _TEXT_CHUNK_SIZE:
            sub_chunks = _split_text(
                content,
                _TEXT_CHUNK_SIZE,
                _TEXT_CHUNK_OVERLAP,
            )

            for sub_chunk in sub_chunks:
                chunks.append(
                    {
                        "content": sub_chunk,
                        "content_type": content_type,
                        "metadata": element["metadata"],
                    }
                )

        else:
            # Tables and images remain atomic.
            chunks.append(element)

    return chunks


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------


def _validate_source_file(file_path: pathlib.Path) -> None:
    """
    Validate that the source document exists and is a supported file.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if not file_path.is_file():
        raise DocumentValidationError(f"Document path is not a file: {file_path}")

    if file_path.stat().st_size == 0:
        raise DocumentValidationError(f"Document is empty: {file_path.name}")

    if file_path.suffix.lower() != ".pdf":
        raise DocumentValidationError(f"Unsupported document type: {file_path.suffix}")


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------


def run_ingestion(file_path: str) -> dict:
    """
    Run the complete ingestion pipeline for a single PDF.

    Pipeline:

        1. Validate source file
        2. Register/update document
        3. Parse PDF with Docling
        4. Prepare chunks
        5. Generate embeddings
        6. Store chunks

    Normal uploads do NOT delete existing documents.

    If ingestion of a brand-new document fails after registration,
    the newly-created document is rolled back.

    If re-ingestion of an existing document fails, the existing
    document/chunks are preserved.
    """

    if not file_path or not str(file_path).strip():
        raise DocumentValidationError("A document path is required.")

    resolved = pathlib.Path(file_path).resolve()

    print(f"[ingestion] Starting ingestion: {resolved}")

    # -----------------------------------------------------------------------
    # Step 1: Validate source
    # -----------------------------------------------------------------------

    _validate_source_file(resolved)

    doc_id = None
    is_new_document = False

    try:

        # -------------------------------------------------------------------
        # Step 2: Register document
        # -------------------------------------------------------------------

        print(f"[ingestion] Registering document: " f"{resolved.name}")

        # The updated db.py will return:
        #
        #     {
        #         "doc_id": "...",
        #         "is_new": True/False
        #     }
        #
        # This lets us safely rollback only a newly-created document.

        document_info = upsert_document(
            resolved.name,
            str(resolved),
        )

        doc_id = document_info["doc_id"]
        is_new_document = document_info["is_new"]

        print(f"[ingestion] doc_id={doc_id} " f"is_new={is_new_document}")

        # -------------------------------------------------------------------
        # Step 3: Parse PDF
        # -------------------------------------------------------------------

        print(f"[ingestion] Parsing document: {resolved}")

        try:
            parsed_elements = parse_document(str(resolved))

        except FileNotFoundError:
            raise

        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse '{resolved.name}'.") from exc

        if not parsed_elements:
            raise DocumentParsingError(
                f"No content could be extracted from " f"'{resolved.name}'."
            )

        print(f"[ingestion] Docling produced " f"{len(parsed_elements)} elements")

        # -------------------------------------------------------------------
        # Step 4: Prepare chunks
        # -------------------------------------------------------------------

        chunks = _prepare_chunks(parsed_elements)

        if not chunks:
            raise DocumentParsingError(
                f"No usable content was extracted from " f"'{resolved.name}'."
            )

        print(f"[ingestion] {len(chunks)} chunks ready " f"for embedding")

        # -------------------------------------------------------------------
        # Step 5 + 6: Embed and store
        # -------------------------------------------------------------------

        try:
            count = store_chunks(
                chunks,
                doc_id,
            )

        except Exception as exc:
            raise DocumentStorageError(
                f"Failed to embed/store chunks for " f"'{resolved.name}'."
            ) from exc

        if count <= 0:
            raise DocumentStorageError(
                f"No chunks were stored for " f"'{resolved.name}'."
            )

        print(f"[ingestion] Stored {count} chunks " f"→ multimodal_chunks")

        # -------------------------------------------------------------------
        # Success
        # -------------------------------------------------------------------

        print(f"[ingestion] Successfully ingested " f"'{resolved.name}'")

        return {
            "status": "success",
            "message": "Document ingested successfully.",
            "doc_id": doc_id,
            "filename": resolved.name,
            "chunks_ingested": count,
        }

    except IngestionError:
        # ---------------------------------------------------------------
        # Our own controlled ingestion errors.
        #
        # Rollback only if this was a newly-created document.
        # Existing documents must survive failed re-ingestion.
        # ---------------------------------------------------------------

        if doc_id and is_new_document:
            _rollback_new_document(
                doc_id,
                resolved.name,
            )

        raise

    except FileNotFoundError:
        if doc_id and is_new_document:
            _rollback_new_document(
                doc_id,
                resolved.name,
            )

        raise

    except Exception as exc:
        # ---------------------------------------------------------------
        # Unexpected failure.
        # ---------------------------------------------------------------

        print(
            f"[ingestion] Unexpected ingestion failure " f"for '{resolved.name}': {exc}"
        )

        if doc_id and is_new_document:
            _rollback_new_document(
                doc_id,
                resolved.name,
            )

        raise IngestionError(
            f"Unexpected failure while ingesting " f"'{resolved.name}'."
        ) from exc


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def _rollback_new_document(
    doc_id: str,
    filename: str,
) -> None:
    """
    Roll back a newly-created document after ingestion failure.

    IMPORTANT:
        This function is intentionally used only for documents that
        did not exist before the current ingestion attempt.

    Existing documents must never be deleted simply because a
    re-ingestion attempt failed.
    """

    try:
        print(f"[ingestion] Rolling back failed ingestion: " f"{filename}")

        delete_document_by_id(doc_id)

        print(f"[ingestion] Rollback completed: " f"{filename}")

    except Exception as rollback_error:
        # Do not hide the original ingestion exception.
        print(
            f"[ingestion] WARNING: rollback failed for "
            f"'{filename}': {rollback_error}"
        )


# ---------------------------------------------------------------------------
# Clear complete ingestion environment
# ---------------------------------------------------------------------------


def clear_ingestion_data() -> dict:
    """
    Completely reset the ingestion environment.

    This operation intentionally removes:

        - documents
        - multimodal_chunks
        - uploaded PDF/DOCX files
        - extracted images

    Unlike normal document ingestion, this operation is destructive.
    """

    print("[ingestion] Starting complete ingestion cleanup...")

    try:
        result = delete_all_ingested_data()

    except Exception as exc:
        print(f"[ingestion] Failed to clear ingestion data: {exc}")

        raise IngestionError("Failed to clear the ingestion environment.") from exc

    print("[ingestion] Complete ingestion cleanup finished.")

    return result


# ---------------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) >= 2:
        pdf_path = pathlib.Path(sys.argv[1])
    else:
        pdf_path = pathlib.Path("data/KB_Credit_Card_Spend_Summarizer.pdf")

    try:
        result = run_ingestion(str(pdf_path))

        print(f"\nIngestion complete: {result}")

    except Exception as exc:
        print(f"\nIngestion failed: {exc}")

        raise

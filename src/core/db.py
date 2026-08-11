import base64
import hashlib
import json
import os
import pathlib

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase

load_dotenv()


class EmbeddingServiceError(RuntimeError):
    """
    Raised when the configured embedding service is unavailable
    or cannot generate embeddings.
    """

    pass


# ===========================================================================
# Configuration
# ===========================================================================

_PG_CONNECTION = os.getenv(
    "PG_CONNECTION_STRING",
    "",
)

_PG_DSN = os.getenv(
    "PG_CONNECTION_STRING_FTS",
)

pg_vector_connection = _PG_CONNECTION

pg_rdbms_connection = os.getenv(
    "PG_RDBMS_CONNECTION_STRING",
)

# ===========================================================================
# Embeddings
# ===========================================================================

_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _create_embedding_client() -> OpenAIEmbeddings:
    """
    Create the OpenAI embedding client after validating configuration.

    This validates local configuration only.
    The actual API/model availability is validated separately by
    validate_embedding_service().
    """

    if not _OPENAI_API_KEY:
        raise EmbeddingServiceError(
            "OPENAI_API_KEY is not configured. " "Please check your .env file."
        )

    if not _EMBED_MODEL:
        raise EmbeddingServiceError(
            "OPENAI_EMBEDDING_MODEL is not configured. " "Please check your .env file."
        )

    try:

        return OpenAIEmbeddings(
            model=_EMBED_MODEL,
            api_key=_OPENAI_API_KEY,
        )

    except Exception as exc:

        print("[db] Failed to initialize embedding client: " f"{exc}")

        raise EmbeddingServiceError(
            "Failed to initialize the OpenAI embedding service."
        ) from exc


_embeddings = _create_embedding_client()


def validate_embedding_service() -> None:
    """
    Perform a real embedding API call to verify that the configured
    embedding model is usable.

    This should be called during application startup.
    """

    if not _OPENAI_API_KEY:
        raise EmbeddingServiceError("OPENAI_API_KEY is not configured.")

    if not _EMBED_MODEL:
        raise EmbeddingServiceError("OPENAI_EMBEDDING_MODEL is not configured.")

    try:

        print("[db] Validating embedding service " f"using model='{_EMBED_MODEL}'...")

        embedding = _embeddings.embed_query("NorthStar embedding health check")

        if not embedding:
            raise EmbeddingServiceError(
                "Embedding service returned an empty embedding."
            )

        print("[db] Embedding service validation successful.")

    except EmbeddingServiceError:
        raise

    except Exception as exc:

        print("[db] Embedding service validation failed: " f"{exc}")

        raise EmbeddingServiceError(
            "The configured OpenAI embedding model is unavailable. "
            f"Model='{_EMBED_MODEL}'. "
            "Check OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY, "
            "model access, API quota, and network connectivity."
        ) from exc


def _embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for a batch of text strings.

    Raises:
        EmbeddingServiceError:
            When the embedding service cannot generate embeddings.
    """

    if not texts:
        return []

    if not _OPENAI_API_KEY:
        raise EmbeddingServiceError("OPENAI_API_KEY is not configured.")

    if not _EMBED_MODEL:
        raise EmbeddingServiceError("OPENAI_EMBEDDING_MODEL is not configured.")

    try:

        embeddings = _embeddings.embed_documents(texts)

        if len(embeddings) != len(texts):

            raise EmbeddingServiceError(
                "Embedding service returned an unexpected number " "of embeddings."
            )

        return embeddings

    except EmbeddingServiceError:
        raise

    except Exception as exc:

        print("[db] Embedding generation failed: " f"{exc}")

        raise EmbeddingServiceError(
            "Failed to generate embeddings using the configured "
            f"OpenAI embedding model '{_EMBED_MODEL}'. "
            "Check the model name, API key, model access, "
            "API quota, and network connectivity."
        ) from exc


# ===========================================================================
# PostgreSQL connection pool
# ===========================================================================

_pool: ConnectionPool | None = None

_sql_database: SQLDatabase | None = None

_sql_schema: str | None = None


def _get_pool() -> ConnectionPool:
    """
    Return the module-level PostgreSQL connection pool.

    The pool is created lazily so the application can start even if the
    database is temporarily unavailable.
    """

    global _pool

    if _pool is not None:
        return _pool

    if not _PG_DSN:
        raise ValueError("PG_CONNECTION_STRING_FTS is not configured.")

    try:

        print("[db] Initializing PostgreSQL connection pool...")

        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={
                "row_factory": dict_row,
            },
        )

        print("[db] PostgreSQL connection pool initialized.")

        return _pool

    except Exception as exc:

        print("[db] Failed to initialize PostgreSQL pool: " f"{exc}")

        raise


def get_db_conn():
    """
    Return a pooled PostgreSQL connection context manager.

    Usage:

        with get_db_conn() as conn:
            with conn.cursor() as cur:
                ...
    """

    try:
        return _get_pool().connection()

    except PoolTimeout as exc:
        print("[db] PostgreSQL connection pool timeout: " f"{exc}")
        raise

    except Exception as exc:
        print("[db] Failed to acquire database connection: " f"{exc}")
        raise


# ===========================================================================
# Document registry
# ===========================================================================


def upsert_document(
    filename: str,
    source_path: str,
) -> dict:
    """
    Insert a document or update an existing document.

    Returns:

        {
            "doc_id": "...",
            "is_new": True/False,
        }

    `is_new` is used by the ingestion layer to determine whether a failed
    ingestion should be rolled back.

    Important:
        Re-ingesting an existing filename does NOT delete the existing
        chunks here. Chunk replacement remains the responsibility of
        store_chunks(), which performs the delete + insert in one transaction.
    """

    if not filename or not filename.strip():
        raise ValueError("Document filename cannot be empty.")

    if not source_path or not source_path.strip():
        raise ValueError("Document source path cannot be empty.")

    try:

        with get_db_conn() as conn:

            with conn.cursor() as cur:

                # ----------------------------------------------------------
                # Determine whether this document already exists.
                # ----------------------------------------------------------

                cur.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE filename = %s
                    """,
                    (filename,),
                )

                existing_row = cur.fetchone()

                is_new = existing_row is None

                # ----------------------------------------------------------
                # Insert or update.
                # ----------------------------------------------------------

                cur.execute(
                    """
                    INSERT INTO documents (
                        filename,
                        source_path
                    )
                    VALUES (
                        %s,
                        %s
                    )
                    ON CONFLICT (filename) DO UPDATE
                    SET
                        source_path = EXCLUDED.source_path,
                        ingested_at = now()
                    RETURNING id
                    """,
                    (
                        filename,
                        source_path,
                    ),
                )

                row = cur.fetchone()

            conn.commit()

        if not row:
            raise RuntimeError(
                "Document upsert completed without returning a document ID."
            )

        doc_id = str(row["id"])

        print(
            f"[db] Document upserted: "
            f"filename={filename}, "
            f"doc_id={doc_id}, "
            f"is_new={is_new}"
        )

        return {
            "doc_id": doc_id,
            "is_new": is_new,
        }

    except Exception as exc:

        print(f"[db] Failed to upsert document " f"'{filename}': {exc}")

        raise


# ===========================================================================
# Delete a single document
# ===========================================================================


def delete_document_by_id(
    doc_id: str,
) -> dict:
    """
    Delete a single document and its associated chunks.

    multimodal_chunks.doc_id references documents.id with ON DELETE CASCADE,
    so deleting the document automatically removes its chunks.

    This function is primarily used to rollback a newly-created document
    when ingestion fails.
    """

    if not doc_id or not str(doc_id).strip():
        raise ValueError("Document ID is required.")

    try:

        with get_db_conn() as conn:

            with conn.cursor() as cur:

                # ----------------------------------------------------------
                # Get document information first so we can remove its
                # filesystem image files as well.
                # ----------------------------------------------------------

                cur.execute(
                    """
                    SELECT filename
                    FROM documents
                    WHERE id = %s::uuid
                    """,
                    (doc_id,),
                )

                document = cur.fetchone()

                if document is None:

                    print(f"[db] Document {doc_id} " "does not exist.")

                    return {
                        "status": "not_found",
                        "doc_id": doc_id,
                        "chunks_deleted": 0,
                        "images_deleted": 0,
                    }

                # ----------------------------------------------------------
                # Delete associated image files before deleting the DB row.
                # ----------------------------------------------------------

                images_deleted = _delete_document_images(doc_id)

                # ----------------------------------------------------------
                # ON DELETE CASCADE removes multimodal_chunks.
                # ----------------------------------------------------------

                cur.execute(
                    """
                    DELETE FROM documents
                    WHERE id = %s::uuid
                    """,
                    (doc_id,),
                )

                documents_deleted = cur.rowcount

            conn.commit()

        print(f"[db] Deleted document {doc_id}; " f"images_deleted={images_deleted}")

        return {
            "status": "success",
            "doc_id": doc_id,
            "documents_deleted": documents_deleted,
            "images_deleted": images_deleted,
        }

    except Exception as exc:

        print(f"[db] Failed to delete document " f"{doc_id}: {exc}")

        raise


# ===========================================================================
# Delete all ingested data
# ===========================================================================


def delete_all_ingested_data() -> dict:
    """
    Delete all ingested documents and associated data.

    Database:
        - Deletes all rows from documents.
        - multimodal_chunks are deleted automatically through
          ON DELETE CASCADE.

    Filesystem:
        - Deletes uploaded PDF/DOCX files from data/.
        - Deletes extracted images from data/images/.

    This is a destructive operation and is intentionally separate from
    normal document ingestion.
    """

    documents_deleted = 0
    source_files_deleted = 0
    images_deleted = 0

    # -----------------------------------------------------------------------
    # Step 1: Database cleanup
    # -----------------------------------------------------------------------

    try:

        with get_db_conn() as conn:

            with conn.cursor() as cur:

                cur.execute("DELETE FROM documents")

                documents_deleted = cur.rowcount

            conn.commit()

        print(f"[db] Deleted {documents_deleted} documents " "and associated chunks.")

    except Exception as exc:

        print("[db] Failed to delete database ingestion data: " f"{exc}")

        # Important:
        # Do not continue with filesystem deletion if the DB cleanup failed.
        #
        # Otherwise we could end up with:
        #
        #     DB still contains documents
        #     files have already been deleted
        #
        # which creates an inconsistent state.

        raise

    # -----------------------------------------------------------------------
    # Step 2: Delete uploaded source documents
    # -----------------------------------------------------------------------

    data_dir = pathlib.Path("data")

    supported_extensions = {
        ".pdf",
        ".docx",
    }

    if data_dir.exists():

        try:

            for file_path in data_dir.iterdir():

                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in supported_extensions:
                    continue

                try:

                    file_path.unlink()

                    source_files_deleted += 1

                except OSError as exc:

                    print("[db] Could not delete source file " f"{file_path}: {exc}")

        except OSError as exc:

            print("[db] Failed to inspect data directory: " f"{exc}")

            raise

    # -----------------------------------------------------------------------
    # Step 3: Delete extracted images
    # -----------------------------------------------------------------------

    image_dir = data_dir / "images"

    if image_dir.exists():

        try:

            for image_file in image_dir.iterdir():

                if not image_file.is_file():
                    continue

                try:

                    image_file.unlink()

                    images_deleted += 1

                except OSError as exc:

                    print("[db] Could not delete image " f"{image_file}: {exc}")

        except OSError as exc:

            print("[db] Failed to inspect image directory: " f"{exc}")

            raise

    # -----------------------------------------------------------------------
    # Result
    # -----------------------------------------------------------------------

    print(
        f"[db] Cleanup complete: "
        f"{documents_deleted} documents, "
        f"{source_files_deleted} source files, "
        f"{images_deleted} images."
    )

    return {
        "documents_deleted": documents_deleted,
        "source_files_deleted": source_files_deleted,
        "images_deleted": images_deleted,
    }


def delete_document(doc_id: str) -> dict:
    """
    Delete one ingested document and all associated data.

    Deletes:
        1. multimodal_chunks belonging to the document
           through ON DELETE CASCADE
        2. documents row
        3. uploaded PDF/DOCX from data/
        4. extracted images belonging to the document

    Args:
        doc_id: UUID of the document.

    Returns:
        Summary of deleted data.
    """

    # ---------------------------------------------------------
    # Step 1: Find the document
    # ---------------------------------------------------------

    with get_db_conn() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT filename, source_path
                FROM documents
                WHERE id = %s::uuid
                """,
                (doc_id,),
            )

            document = cur.fetchone()

    if not document:
        raise ValueError(f"Document with id '{doc_id}' was not found.")

    filename = document["filename"]
    source_path = document["source_path"]

    # ---------------------------------------------------------
    # Step 2: Delete database record
    # ---------------------------------------------------------

    with get_db_conn() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                DELETE FROM documents
                WHERE id = %s::uuid
                """,
                (doc_id,),
            )

            documents_deleted = cur.rowcount

        conn.commit()

    # ---------------------------------------------------------
    # Step 3: Delete source document
    # ---------------------------------------------------------

    source_file_deleted = 0

    if source_path:

        source_file = pathlib.Path(source_path)

        if source_file.exists() and source_file.is_file():

            try:
                source_file.unlink()
                source_file_deleted = 1

            except OSError as exc:
                print(f"[db] Could not delete source file " f"{source_file}: {exc}")

    # ---------------------------------------------------------
    # Step 4: Delete extracted images
    # ---------------------------------------------------------

    images_deleted = _delete_document_images(doc_id)

    print(
        f"[db] Deleted document {doc_id}: "
        f"filename={filename}, "
        f"source_file_deleted={source_file_deleted}, "
        f"images_deleted={images_deleted}"
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "documents_deleted": documents_deleted,
        "source_file_deleted": source_file_deleted,
        "images_deleted": images_deleted,
    }


# ===========================================================================
# Delete images belonging to one document
# ===========================================================================


def _delete_document_images(
    doc_id: str,
) -> int:
    """
    Delete image files generated for a specific document.

    Image filenames are created as:

        {doc_id}_{hash}.png
    """

    image_dir = pathlib.Path("data/images")

    if not image_dir.exists():
        return 0

    deleted = 0

    prefix = f"{doc_id}_"

    try:

        for image_file in image_dir.iterdir():

            if not image_file.is_file():
                continue

            if not image_file.name.startswith(prefix):
                continue

            try:

                image_file.unlink()

                deleted += 1

            except OSError as exc:

                print(f"[db] Could not delete image " f"{image_file}: {exc}")

    except OSError as exc:

        print(f"[db] Failed to inspect image directory: {exc}")

        raise

    return deleted


def list_ingested_documents() -> list[dict]:
    """
    Return all documents currently registered in the knowledge base.

    Returns:
        List containing document ID, filename, source path,
        and ingestion timestamp.
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:

            cur.execute("""
                SELECT
                    id,
                    filename,
                    source_path,
                    ingested_at
                FROM documents
                ORDER BY ingested_at DESC
                """)

            rows = cur.fetchall()

    return [dict(row) for row in rows]


# ===========================================================================
# Chunk storage
# ===========================================================================


def store_chunks(
    chunks: list[dict],
    doc_id: str,
) -> int:
    """
    Embed chunks and store them in multimodal_chunks.

    The existing chunks for this document are deleted and replaced.

    IMPORTANT:
        The DELETE + INSERT operations happen in a single PostgreSQL
        transaction. If anything fails before commit(), PostgreSQL rolls
        the transaction back, preserving the previous chunks.
    """

    if not doc_id or not str(doc_id).strip():
        raise ValueError("Document ID is required.")

    if not chunks:
        return 0

    # -----------------------------------------------------------------------
    # Validate chunks before calling OpenAI.
    # -----------------------------------------------------------------------

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            raise ValueError(f"Invalid chunk at index {index}.")

        content = chunk.get("content")

        if not isinstance(content, str):
            raise ValueError(f"Chunk {index} has invalid content.")

        if not content.strip():
            raise ValueError(f"Chunk {index} contains empty content.")

        if not chunk.get("content_type"):
            raise ValueError(f"Chunk {index} is missing content_type.")

        if not isinstance(
            chunk.get("metadata"),
            dict,
        ):
            raise ValueError(f"Chunk {index} has invalid metadata.")

    # -----------------------------------------------------------------------
    # Generate embeddings.
    # -----------------------------------------------------------------------

    try:

        all_embeddings = _embed_texts([chunk["content"] for chunk in chunks])

    except Exception as exc:

        print("[db] Failed to generate chunk embeddings: " f"{exc}")

        raise

    if len(all_embeddings) != len(chunks):

        raise RuntimeError("Embedding count does not match chunk count.")

    # ---------------------------------------------------------
    # New embeddings were generated successfully.
    # Now it is safe to replace the previous version.
    # ---------------------------------------------------------

    old_images_deleted = _delete_document_images(doc_id)

    print(f"[db] Removed {old_images_deleted} " f"old images for doc_id={doc_id}")

    # -----------------------------------------------------------------------
    # Dedicated columns.
    # -----------------------------------------------------------------------

    _DEDICATED_COLUMNS = {
        "content_type",
        "element_type",
        "section",
        "page_number",
        "source_file",
        "position",
        "image_base64",
    }

    rows_inserted = 0

    try:

        with get_db_conn() as conn:

            with conn.cursor() as cur:

                # ----------------------------------------------------------
                # Verify parent document exists.
                # ----------------------------------------------------------

                cur.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE id = %s::uuid
                    """,
                    (doc_id,),
                )

                if cur.fetchone() is None:

                    raise ValueError(f"Document {doc_id} does not exist.")

                # ----------------------------------------------------------
                # Replace existing chunks.
                #
                # This is inside the same transaction as all inserts.
                # ----------------------------------------------------------

                cur.execute(
                    """
                    DELETE FROM multimodal_chunks
                    WHERE doc_id = %s::uuid
                    """,
                    (doc_id,),
                )

                # ----------------------------------------------------------
                # Insert new chunks.
                # ----------------------------------------------------------

                for chunk, embedding in zip(
                    chunks,
                    all_embeddings,
                ):

                    metadata = chunk["metadata"]

                    image_base64 = metadata.get("image_base64")

                    image_path: str | None = None

                    mime_type = "image/png" if image_base64 else None

                    # ------------------------------------------------------
                    # Save image to filesystem.
                    # ------------------------------------------------------

                    if image_base64:

                        try:

                            image_bytes = base64.b64decode(
                                image_base64,
                                validate=True,
                            )

                            image_dir = pathlib.Path("data/images")

                            image_dir.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

                            image_file = image_dir / f"{doc_id}_{image_hash}.png"

                            image_file.write_bytes(image_bytes)

                            image_path = str(image_file)

                        except Exception as exc:

                            raise RuntimeError(
                                "Failed to save extracted "
                                f"image for document {doc_id}."
                            ) from exc

                    # ------------------------------------------------------
                    # Build pgvector literal.
                    # ------------------------------------------------------

                    embedding_str = (
                        "[" + ",".join(str(value) for value in embedding) + "]"
                    )

                    # ------------------------------------------------------
                    # Keep dedicated fields out of JSONB.
                    # ------------------------------------------------------

                    clean_meta = {
                        key: value
                        for key, value in metadata.items()
                        if key not in _DEDICATED_COLUMNS
                    }

                    cur.execute(
                        """
                        INSERT INTO multimodal_chunks (
                            doc_id,
                            chunk_type,
                            element_type,
                            content,
                            image_path,
                            mime_type,
                            page_number,
                            section,
                            source_file,
                            position,
                            embedding,
                            metadata
                        )
                        VALUES (
                            %s::uuid,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s::jsonb,
                            %s::vector,
                            %s::jsonb
                        )
                        """,
                        (
                            doc_id,
                            chunk["content_type"],
                            metadata.get("element_type"),
                            chunk["content"],
                            image_path,
                            mime_type,
                            metadata.get("page_number"),
                            metadata.get("section"),
                            metadata.get("source_file"),
                            (
                                json.dumps(metadata.get("position"))
                                if metadata.get("position")
                                else None
                            ),
                            embedding_str,
                            json.dumps(clean_meta),
                        ),
                    )

                    rows_inserted += 1

            # --------------------------------------------------------------
            # Commit only after every chunk has been inserted successfully.
            # --------------------------------------------------------------

            conn.commit()

    except Exception as exc:

        print(f"[db] Failed to store chunks for " f"document {doc_id}: {exc}")

        # No explicit rollback is required here.
        #
        # The connection context manager will rollback the failed
        # transaction when the exception leaves the context.
        #
        # This preserves the previous chunks if insertion failed.

        raise

    # -----------------------------------------------------------------------
    # Remove stale image files from a previous successful ingestion.
    #
    # IMPORTANT:
    # Do this only AFTER the new DB transaction commits.
    # -----------------------------------------------------------------------

    # We intentionally don't delete arbitrary image files here because the
    # current DB transaction has already replaced the chunk references.
    #
    # Image cleanup can safely be handled separately during document reset
    # or through a future document-specific cleanup routine.

    print(f"[db] Stored {rows_inserted} chunks " f"for document {doc_id}.")

    return rows_inserted


# ===========================================================================
# Similarity search
# ===========================================================================


def similarity_search(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:
    """
    Find the k most similar chunks to a natural-language query.

    Similarity = 1 - cosine distance.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if chunk_type not in (
        None,
        "text",
        "table",
        "image",
    ):
        raise ValueError("chunk_type must be one of: " "text, table, image.")

    # -----------------------------------------------------------------------
    # Embed query.
    # -----------------------------------------------------------------------

    try:

        query_vec = _embed_texts([query])[0]

    except EmbeddingServiceError:

        print("[db] Vector search unavailable because " "embedding generation failed.")

        raise

    embedding_str = "[" + ",".join(str(value) for value in query_vec) + "]"

    # -----------------------------------------------------------------------
    # Optional chunk-type filter.
    # -----------------------------------------------------------------------

    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content,
            chunk_type,
            page_number,
            section,
            source_file,
            element_type,
            image_path,
            mime_type,
            position,
            metadata,
            1 - (
                embedding <=> %(vec)s::vector
            ) AS similarity
        FROM multimodal_chunks
        WHERE 1 = 1
        {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    try:

        with get_db_conn() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    sql,
                    {
                        "vec": embedding_str,
                        "chunk_type": chunk_type,
                        "k": k,
                    },
                )

                rows = cur.fetchall()

    except Exception as exc:

        print("[db] Vector similarity search failed: " f"{exc}")

        raise

    # -----------------------------------------------------------------------
    # Load images from filesystem.
    # -----------------------------------------------------------------------

    results = []

    for row in rows:

        row = dict(row)

        image_path = row.pop(
            "image_path",
            None,
        )

        if image_path and os.path.exists(image_path):

            try:

                row["image_base64"] = base64.b64encode(
                    pathlib.Path(image_path).read_bytes()
                ).decode()

            except OSError as exc:

                print(f"[db] Failed to read image " f"{image_path}: {exc}")

                row["image_base64"] = None

        else:

            row["image_base64"] = None

        results.append(row)

    return results


# ===========================================================================
# Chunk listing
# ===========================================================================


def get_all_chunks(
    chunk_type: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Return stored chunks, optionally filtered by type.

    This is primarily useful for preview/debugging.
    """

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    if chunk_type not in (
        None,
        "text",
        "table",
        "image",
    ):
        raise ValueError("chunk_type must be one of: " "text, table, image.")

    type_clause = "WHERE chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id,
            content,
            chunk_type,
            page_number,
            section,
            source_file,
            element_type,
            image_path,
            mime_type,
            position,
            metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY
            page_number ASC NULLS LAST,
            id ASC
        LIMIT %(limit)s
    """

    try:

        with get_db_conn() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    sql,
                    {
                        "chunk_type": chunk_type,
                        "limit": limit,
                    },
                )

                rows = cur.fetchall()

    except Exception as exc:

        print("[db] Failed to retrieve chunks: " f"{exc}")

        raise

    results = []

    for row in rows:

        row = dict(row)

        image_path = row.pop(
            "image_path",
            None,
        )

        if image_path and os.path.exists(image_path):

            try:

                row["image_base64"] = base64.b64encode(
                    pathlib.Path(image_path).read_bytes()
                ).decode()

            except OSError as exc:

                print(f"[db] Failed to read image " f"{image_path}: {exc}")

                row["image_base64"] = None

        else:

            row["image_base64"] = None

        results.append(row)

    return results


model = os.getenv("OPENAI_EMBEDDING_MODEL")
api_key = os.getenv("OPENAI_API_KEY")

# ===========================================================================
# Embedding factory
# ===========================================================================


def get_embeddings():
    """
    Return an OpenAI embedding model instance.

    Kept for compatibility with existing code.
    """

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
    )


# ===========================================================================
# Embedding factory
# ===========================================================================


def get_embeddings():
    """
    Return the configured OpenAI embedding model instance.

    Kept for compatibility with existing code.
    """

    return _create_embedding_client()


# ===========================================================================
# LangChain vector search adapter
# ===========================================================================


def search_vector_store(
    query: str,
    k: int = 20,
) -> list[Document]:
    """
    Perform vector similarity search and return LangChain Documents.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    rows = similarity_search(
        query=query,
        k=k,
    )

    documents = []

    for row in rows:

        documents.append(
            Document(
                page_content=row["content"],
                metadata={
                    "source": row.get("source_file"),
                    "page": row.get("page_number"),
                    "section": row.get("section"),
                    "chunk_type": row.get("chunk_type"),
                    "element_type": row.get("element_type"),
                    "similarity": row.get("similarity"),
                    "image_base64": row.get("image_base64"),
                    "mime_type": row.get("mime_type"),
                    "position": row.get("position"),
                    "metadata": row.get("metadata"),
                },
            )
        )

    return documents


# ===========================================================================
# RDBMS SQLDatabase
# ===========================================================================


def get_sql_database() -> SQLDatabase:
    """
    Return the cached SQLDatabase instance.

    The SQLDatabase object is initialized lazily and reused for subsequent
    NL2SQL requests.
    """

    global _sql_database

    if _sql_database is not None:
        return _sql_database

    if not pg_rdbms_connection:
        raise ValueError("PG_RDBMS_CONNECTION_STRING is not set. " "Check your .env.")

    try:

        print("========== Initializing RDBMS SQLDatabase ==========")

        _sql_database = SQLDatabase.from_uri(
            pg_rdbms_connection,
            include_tables=[
                "billing_statements",
                "card_transactions",
                "credit_cards",
                "customers",
                "reward_transactions",
            ],
        )

        print("========== RDBMS SQLDatabase cached ==========")

        return _sql_database

    except Exception as exc:

        print("[db] Failed to initialize SQLDatabase: " f"{exc}")

        raise


# ===========================================================================
# Cached RDBMS schema
# ===========================================================================


def get_cached_schema() -> str:
    """
    Return the cached RDBMS schema.

    The schema is loaded once per application process and reused for
    subsequent NL2SQL requests.
    """

    global _sql_schema

    if _sql_schema is not None:
        return _sql_schema

    try:

        db = get_sql_database()

        print("========== Loading RDBMS schema ==========")

        _sql_schema = db.get_table_info()

        if not _sql_schema:
            raise RuntimeError("RDBMS schema is empty.")

        print("========== RDBMS schema cached ==========")

        return _sql_schema

    except Exception as exc:

        print("[db] Failed to load RDBMS schema: " f"{exc}")

        raise

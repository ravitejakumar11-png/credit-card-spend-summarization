from src.core.db import (
    delete_all_ingested_data,
    delete_document,
    list_ingested_documents,
)

from src.ingestion.ingestion import run_ingestion


def ingest_document(file_path: str) -> dict:
    return run_ingestion(file_path)


def clear_ingested_data() -> dict:
    return delete_all_ingested_data()


def delete_ingested_document(doc_id: str) -> dict:
    return delete_document(doc_id)


def get_ingested_documents() -> list[dict]:
    return list_ingested_documents()

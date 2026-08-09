from src.core.db import delete_all_ingested_data
from src.ingestion.ingestion import run_ingestion


def ingest_document(file_path: str) -> dict:
    return run_ingestion(file_path)


def clear_ingested_data() -> dict:
    return delete_all_ingested_data()
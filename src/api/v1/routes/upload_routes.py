from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.api.v1.services.upload_service import (
    ingest_document,
    clear_ingested_data,
)

router = APIRouter(prefix="/api/v1/upload")


DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF and run the ingestion pipeline.
    """

    # ---------------------------------------------------------
    # Validate file
    # ---------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # ---------------------------------------------------------
    # Save uploaded file
    # ---------------------------------------------------------

    file_path = DATA_DIR / file.filename

    try:
        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        print(f"[upload_route] File uploaded: {file_path}")

        # -----------------------------------------------------
        # Run ingestion
        # -----------------------------------------------------

        result = ingest_document(str(file_path))

        return result

    except Exception as e:

        print(f"[upload_route] Ingestion failed: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.delete("/clear")
def clear_all_ingested_data():
    """
    Delete all ingested documents, chunks and extracted images.
    """

    try:

        result = clear_ingested_data()

        return {
            "status": "success",
            "message": "All ingested data has been cleared.",
            **result,
        }

    except Exception as e:

        print(f"[upload_route] Failed to clear ingested data: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear ingested data: {str(e)}",
        )

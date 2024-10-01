import datetime
from datetime import datetime as dt
from pathlib import Path
import shutil,tempfile
from fastapi import HTTPException, UploadFile
from fastapi import status
from src.config.appconfig import PINECONE_INDEX
from src.ragpipeline.etl import PDFProcessor

async def run_pipeline(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are allowed.",
        )
    try:
        file_dict = save_pdf_to_temp(file)
        documents = await PDFProcessor.process_pdf(file_dict.get("file_location"))
        await PDFProcessor.generate_embeddings(documents,PINECONE_INDEX)
        return {
            "message": "PDF processed and embeddings seeded to vector database successfully.",
            "status":"success",
            "current_time": dt.now().strftime("%x-%X")
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {e}",
        )


def save_pdf_to_temp(file: UploadFile) -> dict:
    """
    Save an uploaded PDF file to a temporary directory that persists after function execution.

    Args:
        file (UploadFile): The PDF file to be saved.

    Returns:
        dict: A dictionary containing the file location and the temp directory path.
    """

    # Create a temporary directory that persists after the function ends
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        file_location = temp_dir / file.filename
        with file_location.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "file_location": str(file_location),
            "temp_dir": str(temp_dir)
        }
    except Exception as e:
        # If an error occurs, remove the temporary directory and re-raise the exception
        shutil.rmtree(temp_dir)
        raise e


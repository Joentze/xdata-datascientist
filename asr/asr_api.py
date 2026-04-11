import io
import logging
import threading
import time
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, UploadFile
from dependencies.logger import get_logger
from api_models.response import PingResponse, ASRResponse

# TODO: ADD LIFESPAN
app = FastAPI()

inference_lock = threading.Lock()


@app.get("/ping", response_model=PingResponse)
def ping(logger: Annotated[logging.Logger, Depends(get_logger)]) -> PingResponse:
    logger.info("receiving ping")
    return PingResponse(message="pong")


@app.post("/asr", response_model=ASRResponse)
def asr_transcribe(file: UploadFile, logger: Annotated[logging.Logger, Depends(get_logger)]) -> ASRResponse:
    logger.info(f"processing file {file.filename} of size {file.size}")
    if file.content_type != "audio/mpeg":
        logger.error("Unsupported file type: %s", file.content_type)
        raise HTTPException(
            status_code=415, detail=f"Unsupported file type: {file.content_type}. Expected audio file.")

    try:
        duration = 0
        transcription = ""

    except Exception as e:
        logger.error("Transcription failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail="Transcription failed.")

    return ASRResponse(transcription=transcription, duration=str(round(duration, 4)))


if __name__ == "__main__":
    import faulthandler
    import uvicorn
    faulthandler.enable()
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")

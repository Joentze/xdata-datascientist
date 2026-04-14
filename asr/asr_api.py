import io
import logging
import threading
import time
from typing import Annotated

import librosa
import numpy as np
import torch
from fastapi import FastAPI, Depends, HTTPException, UploadFile
from dependencies.logger import get_logger
from dependencies.asr_model import ASRModel, get_asr_model
from api_models.response import PingResponse, ASRResponse
from lifespan.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

inference_lock = threading.Lock()


@app.get("/ping", response_model=PingResponse)
def ping(logger: Annotated[logging.Logger, Depends(get_logger)]) -> PingResponse:
    logger.info("receiving ping")
    return PingResponse(message="pong")


@app.post("/asr", response_model=ASRResponse)
def asr_transcribe(
    file: UploadFile,
    logger: Annotated[logging.Logger, Depends(get_logger)],
    asr: Annotated[ASRModel, Depends(get_asr_model)],
) -> ASRResponse:
    logger.info(f"processing file {file.filename} of size {file.size}")
    if file.content_type != "audio/mpeg":
        logger.error("Unsupported file type: %s", file.content_type)
        raise HTTPException(
            status_code=415, detail=f"Unsupported file type: {file.content_type}. Expected audio file.")

    try:
        audio_bytes = file.file.read()

        with inference_lock:
            audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=16_000)

            start = time.perf_counter()
            with torch.no_grad():
                output = asr.model.transcribe(
                    [np.array(audio_array)], timestamps=True
                )
            duration = time.perf_counter() - start

        transcription = output[0].text
        logger.info("Transcription completed in %.4fs for %s", duration, file.filename)

    except Exception as e:
        logger.error("Transcription failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail="Transcription failed.")

    return ASRResponse(transcription=transcription, duration=str(round(duration, 4)))


if __name__ == "__main__":
    import faulthandler
    import uvicorn
    faulthandler.enable()
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")

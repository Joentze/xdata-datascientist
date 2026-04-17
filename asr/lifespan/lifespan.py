import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def get_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    return logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = get_logger()
    app.state.logger = logger

    logger.info("Application starting...")
    logger.info("Loading ASR model (%s) from NeMo", MODEL_NAME)

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info("Using device: %s", device)

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=MODEL_NAME)
        asr_model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 256],
        )
        asr_model = asr_model.to(device)
        asr_model.eval()

        app.state.asr = asr_model
        app.state.device = device
    except Exception as e:
        logger.error("Failed to load ASR model: %s", e, exc_info=True)

    yield

    logger.info("Application stopping...")
    logger.info("Removing model from memory")
    if hasattr(app.state, "asr"):
        del app.state.asr
    if hasattr(app.state, "device"):
        del app.state.device

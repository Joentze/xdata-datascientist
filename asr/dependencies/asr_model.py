from dataclasses import dataclass

import torch
from fastapi import Request, HTTPException

import nemo.collections.asr as nemo_asr


@dataclass
class ASRModel:
    model: nemo_asr.models.ASRModel
    device: torch.device


def get_asr_model(request: Request) -> ASRModel:
    asr = getattr(request.app.state, "asr", None)
    device = getattr(request.app.state, "device", None)
    if asr is None:
        raise HTTPException(
            status_code=503, detail="ASR model is not available")
    return ASRModel(model=asr, device=device)

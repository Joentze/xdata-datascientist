from pydantic import BaseModel


class PingResponse(BaseModel):
    message: str


class ASRResponse(BaseModel):
    """Response (content-type: application/json)

    Attributes:
        transcription: The transcribed text returned by the ASR model.
            e.g., "BEFORE HE HAD TIME TO ANSWER A MUCH
            ENCUMBERED VERA BURST INTO THE ROOM"
        duration: The duration of the file in seconds.
            e.g., "20.7"
    """

    transcription: str
    duration: str

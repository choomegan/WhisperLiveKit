"""
Define Pydantic models for request and response schemas used by the real-time streaming
ASR service.
"""

from typing import List, Optional

from pydantic import BaseModel


class TranscriptionRequest(BaseModel):
    """
    Represents a request from the server. Contains audio which is a base64 encoded PCM
    audio, and initial prompt for whisper transcription.
    """

    audio: str  # base64-encoded PCM audio
    init_prompt: str = ""


class Word(BaseModel):
    """
    Represents a single word with its start and end timestamps.
    """

    word: str
    start: float
    end: float
    probability: float


class SegmentOutput(BaseModel):
    """
    Represents a segment in the transcription output.
    """

    no_speech_prob: float
    words: Optional[List[Word]] = None

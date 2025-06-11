"""
Main server for handling audio trascription requests.
"""

import base64
from typing import List

import numpy as np
from fastapi import FastAPI
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from schema import SegmentOutput, TranscriptionRequest, Word

app = FastAPI()
asr_engine = WhisperModel(
    model_size_or_path="base", device="cuda", compute_type="float16"
)


def encode_output(
    segments: List[Segment],
    info: TranscriptionInfo,
) -> List[SegmentOutput]:
    """
    Encodes the output of a transcription process into a structured response.

    Args:
        segments (List[Segment]):
            A list of transcription segments, each containing text
            and optional word-level information.
        info (TranscriptionInfo):
            Metadata and options related to the transcription,
            including language and output preferences.
    Returns:
        List[SegmentOutput]: An object containing the concatenated outputs.
    """
    out_segments: List[SegmentOutput] = []

    for segment in segments:
        segment_words = None
        if info.transcription_options.word_timestamps and hasattr(segment, "words"):
            segment_words = [
                Word(
                    word=w.word,
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(w.probability),
                )
                for w in segment.words
            ]

        out_segments.append(
            SegmentOutput(
                no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0)),
                words=segment_words,
            )
        )

    return out_segments


@app.post("/transcribe")
def transcribe(request: TranscriptionRequest):
    """
    This endpoint receives an audio file and transcription parameters, processes the 
    audio, performs speech-to-text inference using the loaded model, and returns the
    transcription results in the specified response format.
    """
    # Decode base64 audio
    audio_bytes = base64.b64decode(request.audio)
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)  # Assuming float32 audio

    segments, info = asr_engine.transcribe(
        audio_array,
        language=None,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=True,
        initial_prompt=request.init_prompt,
    )

    segments_list = list(segments)

    response = encode_output(segments_list, info)
    return response

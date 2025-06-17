import sys
import logging
import io
import soundfile as sf
import math
try: 
    import torch
except ImportError: 
    torch = None
from typing import List
import numpy as np
from whisperlivekit.timed_objects import ASRToken
import grpc
from whisperlivekit.whisper_streaming_custom.protos import asr_pb2, asr_pb2_grpc
import base64
import requests
import json

logger = logging.getLogger(__name__)

class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language=lan

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    # def load_model(self, modelsize, cache_dir, model_dir):
    #     raise NotImplementedError("must be implemented in the child class")

    def send_transcription_request(self, audio, init_prompt="", language="auto"):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")



class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def send_transcription_request(self, audio: np.ndarray, init_prompt: str = ""):
        """
        Send transcription request to an asr service
        """
        audio_bytes = audio.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        channel = grpc.insecure_channel("asr-service:50051")
        stub = asr_pb2_grpc.TranscriptionServiceStub(channel)

        request = asr_pb2.TranscriptionRequest(audio_base64=audio_b64, init_prompt=init_prompt, language=self.original_language)
        response = stub.Transcribe(request)

        return response.segments

    
    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word, probability=word.probability)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"
"""
Main server for handling audio trascription requests.
"""

import logging
import base64
import numpy as np
import grpc
from concurrent import futures

from faster_whisper import WhisperModel
from asr_pb2 import Word, SegmentOutput, TranscriptionResponse
import asr_pb2_grpc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


class TranscriptionServiceServicer(asr_pb2_grpc.TranscriptionServiceServicer):
    def __init__(self):
        self.model = WhisperModel(
            "large-v3-turbo", device="cuda", compute_type="float16"
        )

    def Transcribe(self, request, context):
        """
        Processes a transcription request by decoding the base64-encoded audio,
        transcribing it using the WhisperModel, and returning the transcription
        segments.

        Args:
            request: A gRPC request containing the base64-encoded audio and an
                    optional initial prompt for transcription.
            context: gRPC context (not used in this method).

        Returns:
            TranscriptionResponse: A protobuf response containing the transcription
            segments with word-level details and no-speech probabilities.
        """
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        segments, info = self.model.transcribe(
            audio_array,
            language=request.language if request.language != "auto" else None,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            initial_prompt=request.init_prompt,
        )

        pb_segments = []
        sentence = []
        for segment in segments:
            pb_words = []
            if info.transcription_options.word_timestamps and hasattr(segment, "words"):
                for w in segment.words:
                    pb_words.append(
                        Word(
                            word=w.word,
                            start=float(w.start),
                            end=float(w.end),
                            probability=float(w.probability),
                        )
                    )
                    sentence.append(w.word)
            pb_segments.append(
                SegmentOutput(
                    no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0)),
                    words=pb_words,
                )
            )
        logging.info("[Transcription]: %s", "".join(sentence))
        return TranscriptionResponse(segments=pb_segments)


def serve():
    """
    Start the gRPC server and wait for termination.

    The server is set up to listen on port 50051 and runs in an executor with a maximum
    of 10 threads.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    asr_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("gRPC server started on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

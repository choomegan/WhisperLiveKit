"""
Script to mock a WS client and stream data (in WebM format) to the server from an audio
file
"""

import asyncio
import json
import subprocess
import time

import soundfile as sf
import websockets

FILEPATH = "/datasets/AMI/EN2001a/audio/EN2001a.Headset-4.wav"
LOGFILE = "OUTPUTZ_start_176s.txt"
CHUNK_SIZE = 1024  # Tune this to simulate streaming


async def receive_loop(ws, log_file=LOGFILE):
    """
    Loop to receive JSON response from the server. Response contains final text,
    buffers and beg and end time
    """
    last_logged_line = ""
    last_logged_speaker = None
    try:
        while True:
            msg = await ws.recv()
            # print("RECV:", msg)
            data = json.loads(msg)

            lines = data.get("lines", [])
            if not lines:
                continue

            # Get the latest line
            latest_line = lines[-1]
            latest_speaker = latest_line.get("speaker")
            latest_text = latest_line.get("text", "").strip()

            if not latest_text:
                continue

            new_log_line = f"[Speaker {latest_speaker}]: {latest_text}"

            # if same speaker and there is update of the text
            if (
                latest_speaker == last_logged_speaker
                and latest_text != last_logged_line
            ):
                # Overwrite the last line in the file
                try:
                    with open(log_file, "r+", encoding="utf-8") as f:
                        lines = f.readlines()
                        if lines:
                            lines[-1] = new_log_line + "\n"
                            f.seek(0)
                            f.writelines(lines)
                            f.truncate()
                        else:
                            f.write(new_log_line + "\n")
                except FileNotFoundError:
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write(new_log_line + "\n")

            # if different speaker, update text
            elif latest_speaker != last_logged_speaker:
                # Append new speaker's line
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(new_log_line + "\n")

            last_logged_speaker = latest_speaker
            last_logged_line = latest_text

    except websockets.exceptions.ConnectionClosedOK:
        print("Server closed connection cleanly.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")


async def stream_audio_with_ffmpeg(file_path, uri="ws://localhost:8000/asr"):
    """
    Read audio file with soundfile, use ffmpeg to convert the format to webm and stream
    to the server.

    Creates a async task to receive responses from the server concurrently to sending
    audio bytes.
    """
    # Sanity check: Ensure file is 16kHz mono
    info = sf.info(file_path)

    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError("Audio must be 16kHz mono to match backend expectations.")

    print(
        f"Loaded audio with sample rate: {info.samplerate}, channels: {info.channels}"
    )

    # Start FFmpeg to convert PCM/WAV → WebM (Opus)
    ffmpeg_cmd = [
        "ffmpeg",
        # "-ss", '176',           # <-- Seek here: start from a certain duration
        "-i",
        file_path,
        "-f",
        "webm",
        "-c:a",
        "libopus",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket server")

        # Receiving happens concurrently to sending bytes
        recv_task = asyncio.create_task(receive_loop(ws))

        start_time = time.time()
        bytes_sent = 0
        BYTES_PER_SECOND = 32000  # Simulated real-time playback rate (~16kHz mono PCM)

        # Send bytes to the server
        try:
            while True:
                chunk = process.stdout.read(CHUNK_SIZE)
                if not chunk:
                    break

                await ws.send(chunk)
                bytes_sent += len(chunk)

                # NOTE: this is not real-time, it will only be real time for PCM format
                # but we are using WebM
                expected_elapsed = bytes_sent / BYTES_PER_SECOND
                actual_elapsed = time.time() - start_time
                sleep_time = expected_elapsed - actual_elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            print("Finished sending WebM audio.")

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed: {e}")

        finally:
            await ws.close()
            await recv_task


if __name__ == "__main__":
    asyncio.run(stream_audio_with_ffmpeg(FILEPATH))

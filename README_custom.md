## Quick Start with Docker
To start the server and web client, run:
```
docker compose up
```

If you want to change the configurations (e.g. change whisper size, enable vac), edit the `command` in docker-compose.yaml to specify the config.

## Testing
There is a testing script available under `/test`, which creates a websocket client and mocks a stream, without the need to record audio through the microphone in the frontend.

Note that this runs faster than real-time, as we are reading from a .wav file and streaming it using FFmpeg to convert to WebM (Opus) on-the-fly. The data is compressed before being sent over the WebSocket, causing the stream to finish faster than real-time, especially during silences. 


To run the test script:
```
docker exec -it whisperlivekit bash
cd test # you should be in directory /app/test in the container
python3 mock_client_without_buffer.py
```

This will update a text file with the speaker and transcript in the following format:
```
[Speaker 0]: Hello I am megan....
[Speaker 1]: Hello I am not megan...
[Speaker 0]: Hello I am megan again...
```


To view the transcript updates while it is processing, in another terminal you can run:
```
tail -f test/{output_file_name}.txt
```
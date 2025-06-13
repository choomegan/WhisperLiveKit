## Quick Start with Docker
To start the server and web client, run:
```
docker compose up
```

If you want to change the configurations (e.g. change whisper size, enable vac, enable diarization), edit the `command` in docker-compose.yaml to specify the config.

For diarization we are using pyannote models, follow instructions in README.md to accept the the user conditions on huggingface. Create file `hf_token.txt` and place hugginface token there. It will be read as a ENV variable during `docker compose up`.


## Setting up for gRPC
This project connects to the ASR service via gRPC

Copy the proto file from [here](/asr-service/asr.proto) and place it under [this directory](/whisperlivekit/whisper_streaming_custom/protos/). 

The auxillary pb files (e.g. _pb2_grpc.py, _pb2.pyi) are not commited to repository. Please generate them from the .proto files when necessary. 

Run this command 1 directory before `/whisperlivekit`.
```
python3 -m grpc_tools.protoc -I. \
--python_out=. \
--grpc_python_out=. \
whisperlivekit/whisper_streaming_custom/protos/asr.proto
```

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

This testing script does not work for diarization yet.
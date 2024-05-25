#!/bin/bash

TOKENIZER_DIR=$HOME/models/llama-3-8b-instruct
ENGINE_DIR=$HOME/engines/llama-3-8b-instruct

docker run \
    --rm \
    --gpus all \
    -v $TOKENIZER_DIR:/tokenizer \
    -v $ENGINE_DIR:/engine \
    -v $PWD:/app \
    trt-llm \
    /bin/bash -c "python3 /app/test_engine.py"

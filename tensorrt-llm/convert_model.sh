#!/bin/bash

HF_MODEL_DIR=$HOME/models/llama-3-8b-instruct
TRT_MODEL_DIR=$HOME/models/llama-3-8b-instruct-trt

USER_ID=$(id -u)
GROUP_ID=$(id -g)

mkdir -p $TRT_MODEL_DIR

docker run \
    --rm \
    --gpus all \
    --user $USER_ID:$GROUP_ID \
    -v $HF_MODEL_DIR:/input_model \
    -v $TRT_MODEL_DIR:/output_model \
    trt-llm \
    /bin/bash -c "python3 convert_checkpoint.py \
        --model_dir=/input_model \
        --output_dir=/output_model \
        --tp_size=1 \
        --dtype=float16"

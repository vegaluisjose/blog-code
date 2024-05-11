#!/bin/bash

TRT_MODEL_DIR=$HOME/models/llama-3-8b-instruct-trt
TRT_ENGINE_DIR=$HOME/engines/llama-3-8b-instruct

mkdir -p $TRT_ENGINE_DIR

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# --user $USER_ID:$GROUP_ID does not work here for some reason
# therefore, I am adding chown -R $USER_ID:$GROUP_ID /trt-engine at the end
docker run \
    --rm \
    --gpus all \
    -v $TRT_MODEL_DIR:/trt-model \
    -v $TRT_ENGINE_DIR:/trt-engine \
    trt-llm \
    /bin/bash -c "trtllm-build --checkpoint_dir=/trt-model \
        --output_dir=/trt-engine \
        --tp_size=1 \
        --workers=1 \
        --max_batch_size=1 \
        --max_input_len=8192 \
        --max_output_len=8192 \
        --gemm_plugin=float16 \
        --gpt_attention_plugin=float16 && chown -R $USER_ID:$GROUP_ID /trt-engine"

#!/bin/bash

MODELS_DIR=$HOME/models
MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct

huggingface-cli download $MODEL_ID \
    --local-dir $MODELS_DIR/llama-3-8b-instruct \
    --local-dir-use-symlinks False

# base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# install trt-llm dependencies
RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget

# download converter
ARG GIT_SHA=71d8d4d3dc655671f32535d6d2b60cab87f36e87
RUN wget https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/${GIT_SHA}/examples/llama/convert_checkpoint.py -O "/convert_checkpoint.py"

# install trt-llm
RUN pip3 install tensorrt_llm==0.10.0.dev2024042300 -U --pre --extra-index-url https://pypi.nvidia.com

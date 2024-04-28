FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# install dependencies
RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget

# install trt-llm
RUN pip3 install tensorrt_llm==0.10.0.dev2024042300 -U --pre --extra-index-url https://pypi.nvidia.com

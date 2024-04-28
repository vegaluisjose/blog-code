# Llama3 TensorRT-LLM

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /home/vega/models/llama-3-8b-instruct
```

* Change eos_token to `<|eot_id|>`

```bash
docker build trt-llm .
```

```bash
docker run --rm --gpus all -v /home/vega/models/llama-3-8b-instruct:/llama-3-8b-instruct --entrypoint /bin/bash -it trt-llm
```


```bash
wget https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/71d8d4d3dc655671f32535d6d2b60cab87f36e87/examples/llama/convert_checkpoint.py
```

```bash
python3 convert_checkpoint.py --model_dir /llama-3-8b-instruct --output_dir /trt-ckpt --tp_size 1 --dtype float16
```

```bash
trtllm-build --checkpoint_dir /trt-ckpt --output_dir /trt-engine --tp_size 1 --workers 1 --max_batch_size=1 --max_input_len=8000 --max_output_len=8000 --gemm_plugin=float16 --gpt_attention_plugin=float16
```

```bash
python3 run.py
```

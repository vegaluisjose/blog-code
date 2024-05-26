# TensorRT-LLM

## Getting started

* Build container with TensorRT LLM
```bash
docker build trt-llm .
```

* Download HuggingFace model
```bash
bash download.sh
```

* Convert HuggingFace model into TensorRT compatible model
```bash
bash convert_model.sh
```

* Create TensorRT engine from TensorRT model
```bash
bash create_engine.sh
```

* Test TensorRT engine
```bash
bash test_engine.sh
```

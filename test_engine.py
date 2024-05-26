import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer


def generate(user_prompt: str):
    # Get tokenizer from a folder
    tokenizer = AutoTokenizer.from_pretrained("/tokenizer")

    # Llama models do not have a padding token, so we use the EOS token
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # and then we add it from the left, to minimize impact on the output
    tokenizer.padding_side = "left"

    # engine options
    engine_opt = {
        "engine_dir": "/engine",
        "rank": tensorrt_llm.mpi_rank(),
    }

    # engine
    engine = ModelRunner.from_dir(**engine_opt)

    # chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    # tokenize (encode)
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )

    # inference options
    inference_opt = {
        "temperature": 0.1,
        "top_k": 1,
        "repetition_penalty": 1.1,
        "max_new_tokens": 32,
        "end_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.eos_token_id,
    }

    # run inference
    outputs = engine.generate(inputs, **inference_opt)

    # tokenizer (decode)
    # the outputs tensor contains both input and output tokens
    # therefore me need to slice it
    start = inputs.size(-1)
    end = outputs.size(-1)
    text = ""
    for i in range(start, end):
        token = outputs[0][0][i].item()
        # found last token
        if token == tokenizer.eos_token_id:
            break
        text += tokenizer.decode([token])

    return text


if __name__ == "__main__":
    question = "what is 4424+890?"
    answer = generate(question)
    print(f"\n\n{question} -> {answer}")

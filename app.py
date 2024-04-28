import argparse
import uvicorn

from model import ChatCompletionRequest, ChatCompletionResponse
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse


TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()

DEFAULT = """{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo-0125",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
}"""


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/chat/completions")
async def generate(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # Streaming case
    # from typing import AsyncGenerator
    # async def stream_results() -> AsyncGenerator[bytes, None]:
    #     async for request_output in results_generator:
    #         prompt = request_output.prompt
    #         text_outputs = [prompt + output.text for output in request_output.outputs]
    #         ret = {"text": text_outputs}
    #         yield (json.dumps(ret) + "\0").encode("utf-8")

    # if stream:
    #     return StreamingResponse(stream_results())

    return ChatCompletionResponse.model_validate_json(DEFAULT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )

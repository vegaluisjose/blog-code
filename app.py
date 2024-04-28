import argparse
import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/chat/completions")
async def generate(request: Request) -> Response:
    # Streaming case
    # async def stream_results() -> AsyncGenerator[bytes, None]:
    #     async for request_output in results_generator:
    #         prompt = request_output.prompt
    #         text_outputs = [prompt + output.text for output in request_output.outputs]
    #         ret = {"text": text_outputs}
    #         yield (json.dumps(ret) + "\0").encode("utf-8")

    # if stream:
    #     return StreamingResponse(stream_results())

    # Non-streaming case

    return JSONResponse({"text": "hello"})


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

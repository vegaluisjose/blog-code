"""Chat completion related APIs.

Reference: https://platform.openai.com/docs/api-reference/completions
Reference: https://github.com/stillmatic/pydantic-openai/tree/main

"""

from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ChatMessageRole(str, Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"


class ChatCompletionMessage(BaseModel):
    role: ChatMessageRole
    content: str
    name: Optional[str] = Field(None, alias="name")


class ChatCompletionRequest(BaseModel):
    model: str = "model"
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = Field(None, alias="max_tokens")
    temperature: Optional[float] = Field(None, alias="temperature")
    top_p: Optional[float] = Field(None, alias="top_p")
    n: Optional[int] = Field(None, alias="n")
    stream: Optional[bool] = Field(None, alias="stream")
    stop: Optional[List[str]] = Field(None, alias="stop")
    presence_penalty: Optional[float] = Field(None, alias="presence_penalty")
    frequency_penalty: Optional[float] = Field(None, alias="frequency_penalty")
    logit_bias: Optional[Dict[str, int]] = Field(None, alias="logit_bias")
    user: Optional[str] = Field(None, alias="user")


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = Field("stop", alias="finish_reason")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field("chatcmpl-123", alias="id")
    object: str = Field("chat.completion", alias="object")
    created: int = Field(1677652288, alias="created")
    model: str = Field("model", alias="model")
    choices: List[ChatCompletionChoice]
    usage: Usage = Usage()

"""OpenAI-compatible request and response shapes.

This file is GIVEN, complete. The contract's teeth are not the exercise: your
job is to fill in the routes in main.py so they read these requests and return
these responses. Do not weaken these models. The Agentic AI cohort's client
(and the openai Python client) expects exactly these field names.

The `tools`/`tool_choice` fields are accepted from day 1 (the contract says a
consumer's payload always validates) and go unused until the tool-calling
engine at tier 1.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One turn in the conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """The body of POST /v1/chat/completions."""
    model: str
    messages: List[ChatMessage]
    # optional generation controls, with OpenAI-compatible names and defaults.
    # max_tokens has no upper bound here: the reference CLAMPS oversized asks
    # to its MAX_TOKENS setting rather than rejecting them (day 5 sets it).
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    # part of the course contract from day 1 so a consumer's payload always
    # validates; accepted and unused until the tool-calling engine (tier 1).
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None


class ResponseMessage(BaseModel):
    """The assistant message inside a completion choice."""
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    """One completion choice. Week 2 always returns exactly one (index 0)."""
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    """Token accounting. All three must be present and non-negative."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """The body returned by POST /v1/chat/completions (non-streaming)."""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelCard(BaseModel):
    """One entry in GET /v1/models."""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "aidc"


class ModelList(BaseModel):
    """The body returned by GET /v1/models."""
    object: Literal["list"] = "list"
    data: List[ModelCard]


class HealthResponse(BaseModel):
    """The body returned by GET /health."""
    status: Literal["ok"] = "ok"
    model: str

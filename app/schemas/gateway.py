"""Pydantic schemas for OpenAI-compatible chat completion endpoints and gateway payloads."""

import time
import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message schema."""
    role: Literal["system", "user", "assistant", "function"] = Field(
        ..., description="The role of the messages author."
    )
    content: str = Field(..., description="The contents of the message.")
    name: Optional[str] = Field(None, description="The name of the author.")


class ChatCompletionRequest(BaseModel):
    """Request schema for /v1/chat/completions."""
    model: str = Field(..., description="ID of the model to use for completion.")
    messages: List[ChatMessage] = Field(
        ..., min_length=1, description="A list of messages comprising the conversation so far."
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature between 0 and 2."
    )
    max_tokens: Optional[int] = Field(
        default=None, gt=0, description="The maximum number of tokens to generate."
    )
    stream: bool = Field(
        default=False, description="If set, partial message deltas will be sent as Server-Sent Events."
    )
    tenant_id: Optional[str] = Field(
        default="default", description="Tenant or API Key identifier for rate limiting and telemetry."
    )
    skip_cache: bool = Field(
        default=False, description="Force bypass of semantic cache."
    )

    def extract_prompt_text(self) -> str:
        """Extract the canonical prompt text representation from the conversation messages."""
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.messages)


class ChatChoice(BaseModel):
    """Choice representation in standard chat completion."""
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    """Token usage details."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response schema for non-streaming /v1/chat/completions."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    cache_hit: bool = Field(default=False, description="Indicates if the response was served from semantic cache.")


class StreamDelta(BaseModel):
    """Delta payload inside a streaming chunk choice."""
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """Choice inside a streaming chunk."""
    index: int = 0
    delta: StreamDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Server-Sent Event chunk schema for streaming responses."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    redis: str
    embedding_model: str
    version: str

"""LLM client routing requests to OpenAI, Anthropic, and local endpoints with streaming support."""

import logging
from typing import AsyncGenerator, Dict, List, Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class LLMClient:
    """Multi-provider LLM gateway client supporting streaming and model routing."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

        self._openai_client: Optional[AsyncOpenAI] = None
        self._anthropic_client: Optional[AsyncAnthropic] = None

        if openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=openai_api_key)
        if anthropic_api_key:
            self._anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)

    def _is_anthropic_model(self, model: str) -> bool:
        """Check if target model belongs to Anthropic Claude family."""
        return model.lower().startswith("claude")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Execute non-streaming chat completion with provider routing."""
        if self._is_anthropic_model(model):
            return await self._anthropic_completion(messages, model, temperature, max_tokens)
        else:
            return await self._openai_completion(messages, model, temperature, max_tokens)

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token chunks from downstream LLM provider."""
        if self._is_anthropic_model(model):
            async for chunk in self._anthropic_stream(messages, model, temperature, max_tokens):
                yield chunk
        else:
            async for chunk in self._openai_stream(messages, model, temperature, max_tokens):
                yield chunk

    async def _openai_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        """OpenAI-compatible non-streaming completion."""
        client = self._openai_client or AsyncOpenAI(api_key="mock-key")
        kwargs: Dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def _openai_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> AsyncGenerator[str, None]:
        """OpenAI-compatible streaming completion."""
        client = self._openai_client or AsyncOpenAI(api_key="mock-key")
        kwargs: Dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _anthropic_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        """Anthropic Claude non-streaming completion."""
        if not self._anthropic_client:
            raise ValueError("Anthropic API key is not configured.")

        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: Dict = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._anthropic_client.messages.create(**kwargs)
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        return "".join(text_blocks)

    async def _anthropic_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> AsyncGenerator[str, None]:
        """Anthropic Claude streaming completion."""
        if not self._anthropic_client:
            raise ValueError("Anthropic API key is not configured.")

        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: Dict = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with self._anthropic_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

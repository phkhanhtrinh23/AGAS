"""LLM backend adapters (OpenAI and Ollama)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


@dataclass
class LLMRequest:
    """Common request format for coordinator LLM calls."""

    system_prompt: str
    user_prompt: str
    temperature: float = 0.1
    max_tokens: int = 700


class LLMClient(Protocol):
    """Protocol for text generation backends."""

    def generate(self, request: LLMRequest) -> str:
        """Generate one text completion from a normalized request payload.

        Args:
            request: Provider-agnostic generation request payload.

        Returns:
            Generated text response.
        """

        ...


@dataclass
class OpenAIClient:
    """OpenAI client wrapper using the new Responses API."""

    model: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Create the underlying OpenAI SDK client lazily after dataclass init."""

        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key)

    def generate(self, request: LLMRequest) -> str:
        """Call the Responses API and normalize output into plain text.

        Args:
            request: Provider-agnostic generation request payload.

        Returns:
            Generated text extracted from OpenAI response content.
        """
        
        print(f"System prompt:\n{request.system_prompt}\n")
        print(f"User prompt:\n{request.user_prompt}\n")
        
        response = self._client.responses.create(
            model=self.model,
            max_output_tokens=request.max_tokens,
            input=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        )
        text = getattr(response, "output_text", None)
        if text:
            return text

        parts = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                chunk = getattr(content, "text", None)
                if chunk:
                    parts.append(chunk)
        return "\n".join(parts).strip()


@dataclass
class OllamaClient:
    """Local Ollama HTTP API wrapper."""

    model: str
    host: str = "http://localhost:11434"
    timeout_seconds: int = 120

    def generate(self, request: LLMRequest) -> str:
        """Call local Ollama ``/api/generate`` endpoint and return response text.

        Args:
            request: Provider-agnostic generation request payload.

        Returns:
            Generated text returned by Ollama.
        """

        payload = {
            "model": self.model,
            "prompt": f"[SYSTEM]\n{request.system_prompt}\n\n[USER]\n{request.user_prompt}",
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        resp = requests.post(
            f"{self.host.rstrip('/')}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip()


def build_llm_client(provider: str, model: str, api_key: str | None = None, host: str | None = None) -> LLMClient:
    """Factory for LLM clients.

    Args:
        provider: Backend name (``openai`` or ``ollama``).
        model: Model identifier passed to the provider backend.
        api_key: Optional OpenAI API key for ``openai`` provider.
        host: Optional Ollama base URL for ``ollama`` provider.

    Returns:
        Configured LLM client implementation.
    """

    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIClient(model=model, api_key=api_key)
    if provider == "ollama":
        return OllamaClient(model=model, host=host or "http://localhost:11434")
    raise ValueError(f"Unsupported provider: {provider}")

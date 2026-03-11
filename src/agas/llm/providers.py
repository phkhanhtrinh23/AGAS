"""LLM backend adapters (OpenAI and Ollama)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

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


class LLMProviderError(RuntimeError):
    """Raised when a provider backend fails to execute a generation request."""


def _extract_openai_response_text(response: Any) -> str:
    """Extract plain text from a Responses API payload.

    The OpenAI Responses API may return a mix of output item types. Some items
    expose ``content=None`` (for example reasoning/tool-related items), so this
    helper must tolerate missing or non-list content fields.

    Args:
        response: Raw SDK response object returned by ``responses.create``.

    Returns:
        Best-effort plain-text extraction from the response payload.
    """

    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        content_list = getattr(item, "content", None)
        if content_list is None and isinstance(item, dict):
            content_list = item.get("content")
        for content in content_list or []:
            chunk = getattr(content, "text", None)
            if isinstance(chunk, dict):
                chunk = chunk.get("value") or chunk.get("text")
            if chunk is None and isinstance(content, dict):
                chunk = content.get("text") or content.get("output_text")
                if isinstance(chunk, dict):
                    chunk = chunk.get("value") or chunk.get("text")
            if chunk:
                parts.append(str(chunk).strip())
    return "\n".join(part for part in parts if part).strip()


@dataclass
class OpenAIClient:
    """OpenAI client wrapper using the new Responses API."""

    model: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Create the underlying OpenAI SDK client lazily after dataclass init."""
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import error depends on environment
            raise LLMProviderError("Failed to import the OpenAI SDK. Install the 'openai' package.") from exc

        try:
            self._client = OpenAI(api_key=self.api_key)
        except Exception as exc:
            raise LLMProviderError(f"Failed to initialize OpenAI client: {exc}") from exc

    def generate(self, request: LLMRequest) -> str:
        """Call the Responses API and normalize output into plain text.

        Args:
            request: Provider-agnostic generation request payload.

        Returns:
            Generated text extracted from OpenAI response content.
        """
        try:
            response = self._client.responses.create(
                model=self.model,
                max_output_tokens=request.max_tokens,
                input=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
            )
        except Exception as exc:
            raise LLMProviderError(f"OpenAI request failed for model '{self.model}': {exc}") from exc
        return _extract_openai_response_text(response)


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
        try:
            resp = requests.post(
                f"{self.host.rstrip('/')}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise LLMProviderError(f"Ollama request failed for model '{self.model}': {exc}") from exc
        except ValueError as exc:
            raise LLMProviderError(f"Ollama returned invalid JSON for model '{self.model}': {exc}") from exc
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

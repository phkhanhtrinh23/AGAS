"""LLM backend adapters.

The paper's headline configuration uses the **OpenAI API** as the LLM backbone
for both the Coordinator and the worker policies. The :class:`RuleBasedProvider`
is a minimal deterministic fallback used when ``OPENAI_API_KEY`` is missing —
this keeps the smoke tests and the public demo runnable without network access
or a paid API key.

Provider selection order at runtime
-----------------------------------
1. ``provider="openai"`` (default) with a non-empty ``OPENAI_API_KEY`` → use
   :class:`OpenAIClient` with the model from ``OPENAI_MODEL`` (default
   ``gpt-5.1``).
2. ``provider="rule"`` or ``OPENAI_API_KEY`` missing → fall back to
   :class:`RuleBasedProvider` (emits an empty JSON action list; the worker /
   coordinator then drops back to its own rule-based policy).
3. ``provider="ollama"`` is also supported for local experiments.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import requests

_logger = logging.getLogger(__name__)


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


@dataclass
class RuleBasedProvider:
    """Deterministic fallback provider.

    This client returns an empty JSON action payload so that the calling
    Coordinator / Worker immediately falls back to its own rule-based logic
    (the rule fallback is implemented inside :mod:`agas.agents.worker` and
    :mod:`agas.agents.coordinator`). It is the provider used when no
    ``OPENAI_API_KEY`` is configured.

    Returning an empty actions list is deliberate: it makes the rule fallback
    the *single* deterministic policy when LLM access is unavailable, instead
    of trying to imitate an LLM response.
    """

    model: str = "rule-based"

    def generate(self, request: LLMRequest) -> str:  # noqa: D401 - protocol impl
        """Always return an empty action list to force the rule fallback path."""

        return '{"actions": [], "strategy": null, "rationale": "rule-based fallback"}'


def build_llm_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    host: str | None = None,
) -> LLMClient:
    """Factory for LLM clients with paper-aligned defaults.

    Defaults
    --------
    * ``provider`` defaults to ``"openai"`` (the paper backbone).
    * ``model`` defaults to the value of the ``OPENAI_MODEL`` environment
      variable, falling back to ``"gpt-5.1"`` if unset.
    * When ``OPENAI_API_KEY`` is missing the function emits a single warning
      and returns a :class:`RuleBasedProvider`, regardless of ``provider``.

    Args:
        provider: ``"openai"`` (default), ``"ollama"`` or ``"rule"``.
        model: Model identifier passed to the chosen backend.
        api_key: OpenAI API key; defaults to ``OPENAI_API_KEY`` env var.
        host: Optional Ollama base URL.

    Returns:
        Configured LLM client implementation.
    """

    provider = (provider or "openai").lower().strip()
    resolved_model = model or os.environ.get("OPENAI_MODEL", "gpt-5.1")
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")

    if provider == "rule":
        return RuleBasedProvider(model=resolved_model)

    if provider == "openai":
        if not resolved_key:
            _logger.warning(
                "OPENAI_API_KEY is not set. Falling back to RuleBasedProvider — "
                "results will be deterministic and not LLM-driven."
            )
            return RuleBasedProvider(model=resolved_model)
        return OpenAIClient(model=resolved_model, api_key=resolved_key)

    if provider == "ollama":
        return OllamaClient(model=resolved_model, host=host or "http://localhost:11434")

    raise ValueError(f"Unsupported provider: {provider}")

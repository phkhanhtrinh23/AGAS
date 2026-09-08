"""Tests for LLM provider response normalization helpers."""

import os

import pytest

from agas.llm.providers import (
    LLMProviderError,
    LLMRequest,
    OpenAIClient,
    RuleBasedProvider,
    _extract_openai_response_text,
    build_llm_client,
)


class _Obj:
    """Minimal attribute container for provider-shape test doubles."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_extract_openai_response_text_skips_none_content_items() -> None:
    """Ensure OpenAI response parsing tolerates output items with ``content=None``."""

    response = _Obj(
        output_text=None,
        output=[
            _Obj(type="reasoning", content=None),
            _Obj(type="message", content=[_Obj(text="first line"), _Obj(text="second line")]),
        ],
    )

    assert _extract_openai_response_text(response) == "first line\nsecond line"


def test_extract_openai_response_text_supports_dict_content_shapes() -> None:
    """Ensure text extraction works when the SDK surfaces dict-like content payloads."""

    response = _Obj(
        output_text=None,
        output=[
            {"content": [{"text": {"value": "json line"}}]},
        ],
    )

    assert _extract_openai_response_text(response) == "json line"


def test_openai_client_wraps_provider_errors() -> None:
    """Ensure OpenAI provider failures are raised as normalized ``LLMProviderError`` exceptions."""

    class _Responses:
        def create(self, **kwargs):  # pragma: no cover - simple failure shim
            raise RuntimeError("network down")

    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "gpt-test"
    client.api_key = None
    client._client = _Obj(responses=_Responses())

    with pytest.raises(LLMProviderError, match="OpenAI request failed"):
        client.generate(LLMRequest(system_prompt="s", user_prompt="u"))


def test_rule_based_provider_returns_empty_action_list() -> None:
    """RuleBasedProvider always returns an empty actions JSON so the host
    rule-fallback path fires deterministically."""

    provider = RuleBasedProvider()
    response = provider.generate(LLMRequest(system_prompt="s", user_prompt="u"))
    assert "actions" in response
    # Empty action list is the deliberate contract.
    assert '"actions": []' in response or '"actions":[]' in response


def test_build_llm_client_falls_back_to_rule_when_api_key_missing(monkeypatch) -> None:
    """When OPENAI_API_KEY is unset, build_llm_client must return a
    RuleBasedProvider (with a warning)."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = build_llm_client(provider="openai", api_key=None)
    assert isinstance(client, RuleBasedProvider)


def test_build_llm_client_explicit_rule_provider() -> None:
    """`provider="rule"` must always return a RuleBasedProvider regardless of
    environment configuration."""

    client = build_llm_client(provider="rule")
    assert isinstance(client, RuleBasedProvider)

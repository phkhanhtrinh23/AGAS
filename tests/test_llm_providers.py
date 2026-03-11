"""Tests for LLM provider response normalization helpers."""

import pytest

from agas.llm.providers import LLMProviderError, LLMRequest, OpenAIClient, _extract_openai_response_text


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

"""
Provider wiring, checkable without a key or a network call.

Constructing a model only builds the SDK client, so these assert on how it was
configured rather than on any request.
"""

import pytest

from agent import DeepSeekModel, OpenAIModel, _messages_have_image, create_model


def test_deepseek_reuses_the_openai_path():
    model = create_model("deepseek", "test-key")
    # A subclass of OpenAIModel, so it inherits the json_schema-with-fallback
    # generate() rather than duplicating it.
    assert isinstance(model, OpenAIModel)
    assert isinstance(model, DeepSeekModel)


def test_deepseek_points_at_deepseek():
    model = create_model("deepseek", "test-key")
    assert model.model == "deepseek-v4-pro"
    assert "api.deepseek.com" in str(model.client.base_url)


def test_openai_keeps_its_own_host():
    model = create_model("openai", "test-key")
    assert "api.deepseek.com" not in str(model.client.base_url)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        create_model("nope", "test-key")


def test_model_override_applies_per_provider():
    assert create_model("deepseek", "k", "deepseek-v4-flash").model == "deepseek-v4-flash"
    assert create_model("openai", "k", "gpt-4o-mini").model == "gpt-4o-mini"
    assert create_model("claude", "k", "claude-opus-4-8").model == "claude-opus-4-8"


def test_deepseek_override_still_points_at_deepseek():
    model = create_model("deepseek", "k", "deepseek-v4-flash")
    assert "api.deepseek.com" in str(model.client.base_url)


def test_none_model_keeps_the_provider_default():
    assert create_model("deepseek", "k").model == "deepseek-v4-pro"
    assert create_model("openai", "k").model == "gpt-4o"


def test_deepseek_uses_json_object_mode():
    # DeepSeek rejects json_schema; json_object is its JSON mode.
    assert create_model("deepseek", "k")._structured_format() == {"type": "json_object"}
    assert create_model("openai", "k")._structured_format()["type"] == "json_schema"


def test_deepseek_extends_the_system_prompt_with_allowed_values():
    prepared = create_model("deepseek", "k")._prepare_system("BASE")
    assert prepared.startswith("BASE")
    assert "json" in prepared.lower()  # json_object mode requires the word
    assert "'html' or 'screenshot'" in prepared


def test_deepseek_refuses_image_turns_without_a_network_call():
    image_msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
        ],
    }]
    assert _messages_have_image(image_msg) is True
    with pytest.raises(ValueError, match="screenshot mode is unavailable"):
        create_model("deepseek", "k").generate(image_msg, "sys")


def test_text_only_messages_are_not_flagged_as_images():
    assert _messages_have_image([{"role": "user", "content": "plain text"}]) is False

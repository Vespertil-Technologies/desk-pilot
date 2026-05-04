from agent import _to_openai_messages


def test_string_content_passes_through():
    msgs = [{"role": "user", "content": "hello"}]
    assert _to_openai_messages(msgs) == [{"role": "user", "content": "hello"}]


def test_text_block_kept_as_text():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert _to_openai_messages(msgs) == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]


def test_image_block_becomes_image_url():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "ABCDEF",
                    },
                },
            ],
        }
    ]
    assert _to_openai_messages(msgs) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ABCDEF"},
                },
            ],
        }
    ]


def test_role_is_preserved():
    msgs = [{"role": "assistant", "content": "ok"}]
    assert _to_openai_messages(msgs)[0]["role"] == "assistant"

import pytest

from agent import _extract_json


def test_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_with_markdown_json_fence():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_with_bare_fence():
    assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_with_surrounding_prose():
    assert _extract_json('Here it is: {"a": 1} and that\'s all') == {"a": 1}


def test_nested_object():
    assert _extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}


def test_multiline_json():
    raw = """
    Some intro.
    {
        "mode": "html",
        "action": "click",
        "selector": "#go"
    }
    """
    assert _extract_json(raw)["selector"] == "#go"


def test_no_json_raises():
    with pytest.raises(ValueError):
        _extract_json("nothing here")

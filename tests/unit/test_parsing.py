from agent.parsing import parse_json


def test_pure_json():
    assert parse_json('{"ok": true}') == {"ok": True}


def test_nested_json():
    assert parse_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}


def test_non_object_list_returns_none():
    assert parse_json('[1, 2]') is None


def test_invalid_returns_none():
    assert parse_json("not json at all") is None


def test_prose_wrapped_treated_as_error():
    assert parse_json('Here is the result: {"ok": true} end.') is None


def test_empty_and_none_returns_none():
    assert parse_json("") is None
    assert parse_json(None) is None


def test_whitespace_only_returns_none():
    assert parse_json("   ") is None
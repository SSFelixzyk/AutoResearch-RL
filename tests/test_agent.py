# tests/test_agent.py
from unittest.mock import patch, MagicMock
from src.agent import extract_json, LLMAgent

def test_extract_json_valid():
    text = 'some text {"key": "value"} more text'
    result = extract_json(text)
    assert result == {"key": "value"}

def test_extract_json_invalid_returns_none():
    result = extract_json("no json here")
    assert result is None

def test_extract_json_fenced():
    text = '```json\n{"a": 1}\n```'
    result = extract_json(text)
    assert result == {"a": 1}

def test_extract_json_nested():
    text = '{"outer": {"inner": 42}}'
    result = extract_json(text)
    assert result == {"outer": {"inner": 42}}

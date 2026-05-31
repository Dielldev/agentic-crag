"""Tests for extract_json (app/llm_client.py) — tolerant JSON parsing of LLM output."""

import pytest

from app.llm_client import extract_json


def test_plain_json():
    assert extract_json('{"relevance": 80, "completeness": 70}') == {
        "relevance": 80,
        "completeness": 70,
    }


def test_json_fenced():
    raw = '```json\n{"grounded": true, "confidence": 90}\n```'
    assert extract_json(raw) == {"grounded": True, "confidence": 90}


def test_bare_fence():
    raw = '```\n{"a": 1}\n```'
    assert extract_json(raw) == {"a": 1}


def test_prose_wrapped():
    raw = 'Here is the result: {"relevance": 55} — hope that helps!'
    assert extract_json(raw) == {"relevance": 55}


def test_invalid_raises():
    with pytest.raises(Exception):
        extract_json("there is no json object here")

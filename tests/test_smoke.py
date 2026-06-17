from __future__ import annotations


def test_settings_load():
    from src.settings import get_settings
    s = get_settings()
    assert s.nim_base_url.startswith("https://")
    assert s.max_tokens > 0


def test_zero_shot_builds_messages():
    from src.prompts import zero_shot
    task = {"system_prompt": "Classify sentiment.", "examples": []}
    sample = {"input": "Great product!", "expected": "positive"}
    msgs = zero_shot(task, sample)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"


def test_few_shot_includes_examples():
    from src.prompts import few_shot
    task = {
        "system_prompt": "Classify sentiment.",
        "examples": [{"input": "Good", "output": "positive"}],
    }
    sample = {"input": "Bad product", "expected": "negative"}
    msgs = few_shot(task, sample)
    roles = [m["role"] for m in msgs]
    assert "assistant" in roles


def test_cot_appends_instruction():
    from src.prompts import chain_of_thought
    task = {"system_prompt": "Classify sentiment.", "examples": []}
    sample = {"input": "Okay product.", "expected": "neutral"}
    msgs = chain_of_thought(task, sample)
    assert "step by step" in msgs[0]["content"]


def test_strategies_keys():
    from src.prompts import STRATEGIES
    assert set(STRATEGIES.keys()) == {"zero_shot", "few_shot", "chain_of_thought"}

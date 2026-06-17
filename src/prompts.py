"""Prompt builders for the three strategies being compared.

Each builder takes a task definition dict and a sample dict and returns
a list of messages ready to send to the chat model.
"""
from __future__ import annotations


def zero_shot(task: dict, sample: dict) -> list[dict]:
    return [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": sample["input"]},
    ]


def few_shot(task: dict, sample: dict) -> list[dict]:
    messages = [{"role": "system", "content": task["system_prompt"]}]
    for ex in task.get("examples", []):
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": sample["input"]})
    return messages


def chain_of_thought(task: dict, sample: dict) -> list[dict]:
    cot_system = (
        task["system_prompt"]
        + "\nThink step by step before giving your final answer. "
        "End your response with: 'Answer: <your final answer>'"
    )
    return [
        {"role": "system", "content": cot_system},
        {"role": "user", "content": sample["input"]},
    ]


STRATEGIES = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
}

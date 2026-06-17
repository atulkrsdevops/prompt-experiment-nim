"""LLM-as-judge scorer.

Grades a model response against the expected answer.
Returns 1 (correct) or 0 (incorrect).
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import get_chat_model

JUDGE_SYSTEM = (
    "You are grading an AI response against a reference answer. "
    "Reply with only 'correct' or 'incorrect'. "
    "The response is correct if it conveys the same meaning, even if worded differently."
)


def score(question: str, response: str, expected: str) -> int:
    model = get_chat_model(temperature=0)
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"Question: {question}\nExpected: {expected}\nResponse: {response}"),
    ]
    verdict = model.invoke(messages).content.strip().lower()
    return 1 if verdict.startswith("correct") else 0

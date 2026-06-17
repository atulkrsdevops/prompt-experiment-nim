"""Experiment runner.

Loads a task definition JSON, runs each strategy against all samples,
scores responses, and saves results to results/.

Usage:
    python -m src.run --experiment experiments/classification.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import get_chat_model
from .prompts import STRATEGIES
from .scorer import score
from .settings import get_settings


def _messages_to_langchain(messages: list[dict]):
    out = []
    for m in messages:
        if m["role"] == "system":
            out.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(content=m["content"]))
    return out


def run_experiment(experiment_path: str) -> dict:
    task = json.loads(pathlib.Path(experiment_path).read_text())
    s = get_settings()
    model = get_chat_model()

    print(f"\nRunning experiment: {task['name']}")
    print(f"Model: {s.chat_model}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Samples: {len(task['samples'])}\n")

    results = {"experiment": task["name"], "model": s.chat_model, "strategies": {}}

    for strategy_name, builder in STRATEGIES.items():
        scores = []
        latencies = []
        responses = []

        for sample in task["samples"]:
            messages = _messages_to_langchain(builder(task, sample))
            t0 = time.perf_counter()
            response = model.invoke(messages).content.strip()
            latency_ms = (time.perf_counter() - t0) * 1000

            s_score = score(sample["input"], response, sample["expected"])
            scores.append(s_score)
            latencies.append(latency_ms)
            responses.append(response)

        total = len(scores)
        correct = sum(scores)
        avg_lat = sum(latencies) / total

        results["strategies"][strategy_name] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "avg_latency_ms": round(avg_lat),
            "responses": responses,
        }

        print(f"{strategy_name:<20} {correct}/{total}  avg {avg_lat:.0f}ms")

    # save results
    out_dir = pathlib.Path(s.results_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{task['name']}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Path to experiment JSON")
    args = parser.parse_args()
    run_experiment(args.experiment)


if __name__ == "__main__":
    main()

"""Reporter: load all saved results and print a comparison table.

Usage:
    python -m src.report
"""
from __future__ import annotations

import json
import pathlib

from .settings import get_settings


def print_table(results: dict) -> None:
    print(f"\nExperiment: {results['experiment']}  |  Model: {results['model']}")
    print(f"{'strategy':<22} {'score':<8} {'accuracy':<10} {'avg_latency_ms'}")
    print("-" * 60)
    for name, data in results["strategies"].items():
        print(
            f"{name:<22} "
            f"{data['correct']}/{data['total']}  "
            f"{data['accuracy']:.0%}       "
            f"{data['avg_latency_ms']}ms"
        )


def main() -> None:
    results_dir = pathlib.Path(get_settings().results_dir)
    files = sorted(results_dir.glob("*.json"))
    if not files:
        print("No results found. Run an experiment first: python -m src.run --experiment experiments/classification.json")
        return
    for path in files:
        print_table(json.loads(path.read_text()))
    print()


if __name__ == "__main__":
    main()

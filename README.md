# Prompt Experiment Harness on NVIDIA NIM

A reproducible framework for comparing **prompting strategies** — zero-shot,
few-shot, and chain-of-thought — across tasks, with automated scoring tracked
in a results log.

```
Task definition (experiments/*.json)
   |
   v
[Runner]  -- sends each prompt strategy to NIM
   |
   v
[Scorer]  -- grades responses with LLM-as-judge
   |
   v
[Reporter] -- prints comparison table + saves results/
```

Covers **NCA-GENL**: Prompt Engineering, Experiment Design, Data Analysis,
Evaluation & Alignment.

> Free to run on hosted NIM endpoints. No GPU required.

---

## Demo

![Experiment runs and reporter output](docs/screenshots/results.jpg)

---

## Results

### sentiment_classification

```text
strategy           score    accuracy    avg_latency_ms
-----------------  -------  ----------  ----------------
zero_shot          5/5      100%        1016ms
few_shot           5/5      100%        787ms
chain_of_thought   5/5      100%        890ms
```

### question_answering

```text
strategy           score    accuracy    avg_latency_ms
-----------------  -------  ----------  ----------------
zero_shot          5/5      100%        1107ms
few_shot           5/5      100%        985ms
chain_of_thought   5/5      100%        985ms
```

**Key finding:** Few-shot matches chain-of-thought accuracy at lower latency on
both tasks. CoT adds 100-300ms overhead with no accuracy gain here — optimal
strategy depends on task type and latency budget.

---

## Quickstart

```bash
# Windows
copy .env.example .env
pip install -r requirements.txt
python -m src.run --experiment experiments/classification.json
python -m src.run --experiment experiments/qa.json
python -m src.reporter

# Mac/Linux
cp .env.example .env
make setup
make run-classification
make run-qa
make report
```

---

## Adding your own experiment

Create a JSON file in `experiments/` following this structure:

```json
{
  "name": "my_task",
  "system_prompt": "Your task instruction here.",
  "examples": [
    { "input": "example input", "output": "example output" }
  ],
  "samples": [
    { "input": "test input", "expected": "expected answer" }
  ]
}
```

Then run:

```bash
python -m src.run --experiment experiments/my_task.json
```

---

## How it maps to the exam blueprints

| Component | File | NCA-GENL domain |
|---|---|---|
| Zero-shot / few-shot / CoT builders | `src/prompts.py` | Prompt engineering |
| Experiment runner | `src/run.py` | Experiment design |
| LLM-as-judge scorer | `src/scorer.py` | Evaluation |
| Reporter with comparison table | `src/reporter.py` | Data analysis |
| JSON experiment definitions | `experiments/` | Reproducibility |
| CI pipeline | `.github/workflows/` | Software development |

---

## Project structure

```
prompt-experiment-nim/
├── src/
│   ├── settings.py      # env-driven config
│   ├── llm.py           # ChatNVIDIA factory
│   ├── runner.py        # runs all strategies for an experiment
│   ├── scorer.py        # LLM-as-judge scoring
│   ├── reporter.py      # prints comparison tables
│   └── prompts.py       # zero-shot, few-shot, CoT prompt builders
├── experiments/         # task definitions (JSON)
├── results/             # auto-saved run outputs (gitignored)
├── tests/               # offline smoke tests
└── .github/workflows/   # CI pipeline
```

## License

MIT

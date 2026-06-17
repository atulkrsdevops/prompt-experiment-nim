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

```text
$ python -m src.run --experiment experiments/classification.json

Running experiment: sentiment_classification
Model: meta/llama-3.1-8b-instruct
Strategies: zero_shot, few_shot, chain_of_thought
Samples: 5

strategy           score    avg_latency_ms
-----------------  -------  ----------------
zero_shot          3/5      843
few_shot           5/5      921
chain_of_thought   5/5      1243
```

---

## Results

```text
Experiment: sentiment_classification
zero_shot        : 3/5  (60%)  avg 843ms
few_shot         : 5/5  (100%) avg 921ms
chain_of_thought : 5/5  (100%) avg 1243ms

Experiment: question_answering
zero_shot        : 4/5  (80%)  avg 761ms
few_shot         : 5/5  (100%) avg 889ms
chain_of_thought : 5/5  (100%) avg 1311ms
```

**Key finding:** Few-shot matches CoT accuracy at lower latency — optimal
strategy depends on task type and latency budget.

---

## Quickstart

```bash
# Windows
copy .env.example .env
pip install -r requirements.txt
python -m src.run --experiment experiments/classification.json
python -m src.run --experiment experiments/qa.json
python -m src.report
```

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

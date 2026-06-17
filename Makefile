.PHONY: setup run-classification run-qa report test lint

setup:
	pip install -r requirements.txt

run-classification:
	python -m src.run --experiment experiments/classification.json

run-qa:
	python -m src.run --experiment experiments/qa.json

report:
	python -m src.reporter

test:
	pytest -q

lint:
	ruff check .

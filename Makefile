.PHONY: install dev test test-cov lint typecheck format clean run check lock ingest validate train evaluate tune explain docker-build docker-up docker-down study-pdf study-epub study

install:
	uv sync --no-dev

dev:
	uv sync

lock:
	uv lock

test:
	uv run pytest tests/

test-cov:
	uv run pytest tests/ --cov=src --cov-report=html

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

ingest:
	uv run python -m src.data.ingest

validate:
	uv run python -m src.data.validation

train:
	uv run python -m src.model.train

tune:
	uv run python -m src.model.tune

evaluate:
	uv run python -m src.model.evaluate

explain:
	uv run python -m src.explainability.explain

run:
	uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t credit-risk-api:local .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info/

study-pdf:
	uv run study_pdf/build.py

study-epub:
	uv run study_pdf/build_epub.py

study: study-pdf study-epub

check: lint typecheck test

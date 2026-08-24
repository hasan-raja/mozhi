.PHONY: api worker lint fmt typecheck test ci

api:
	uv run uvicorn app.asgi:app --reload --port 8000

worker:
	uv run celery -A app.tasks.celery_app worker -Q extract,vad,asr,translate,tts,qc,stitch -l info

lint:
	uv run ruff check app tests

fmt:
	uv run ruff check --fix app tests && uv run ruff format app tests

typecheck:
	uv run mypy app

test:
	uv run pytest -n auto

ci: lint typecheck test

.PHONY: api worker lint fmt typecheck test ci

api:
	UV_LINK_MODE=copy uv run uvicorn app.asgi:app --reload --port 8000

worker:
	UV_LINK_MODE=copy uv run celery -A app.celery_app.celery_app worker -Q extract,vad,asr,translate,tts,qc,stitch -l info

lint:
	uv run ruff check app tests

fmt:
	uv run ruff check --fix app tests && uv run ruff format app tests

typecheck:
	uv run mypy app

test:
	uv run pytest -n auto

ci: lint typecheck test

# WSL note: the repo lives on /mnt/f (Windows drive). uv can't hardlink there,
# and a stale .venv/lib64 symlink breaks every run — hence UV_LINK_MODE=copy
# and the lib64 cleanup guard below.
preflight:
	rm -f .venv/lib64 .venv/lib 2>/dev/null || true

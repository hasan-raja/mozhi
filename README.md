# Mozhi (மொழி)

AI multilingual dubbing platform — upload a video in one language, get it dubbed in another.
Portfolio-grade miniature of a production media pipeline: extract → VAD → ASR → translate → TTS → QC → stitch.

**Why:** to prove, in code, the exact class of system Sarvam Studio runs — fault-tolerant
orchestration of ML pipelines at scale. See `PLAN.md` for the 7-day build with per-step rationale.

## Architecture

- **API**: FastAPI (async), app-factory pattern
- **Data**: PostgreSQL + async SQLAlchemy 2.0 + Alembic migrations
- **Tasks**: Celery with per-stage queues (extract / vad / asr / translate / tts / qc / stitch), KEDA autoscaling
- **Reliability**: idempotent stages, retries w/ backoff+jitter, stuck-job reaper, DLQ, 429 backpressure at ingress
- **Real-time**: WebSocket job tracking (Redis pub/sub), SSE live translation
- **Engines** (pluggable): Sarvam API (primary) · OpenRouter (translation fallback) · local open models · MockEngine (CI)
- **SDK**: `mozhi-sdk` — auth, rate limiting, metering, OpenTelemetry across FastAPI→Celery
- **Deploy**: Docker multi-role image, Helm chart on k3s; k6 load-tested
- **UI**: Mozhi Studio console (React/Vite/TS) — live pipeline progress, side-by-side dub preview

## Status

🚧 Week-1 build in progress — see PLAN.md.

## Quickstart

```bash
docker compose up -d        # postgres + redis
uv sync                     # deps
alembic upgrade head
make api                    # FastAPI on :8000
make worker                 # Celery workers
```

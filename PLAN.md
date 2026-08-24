# Mozhi — 7-Day Heavy Build Plan (v2, Aug 2026)

Target: **Sarvam Backend Engineer – Studio** (+ Bolna). Every step = one or more MICRO COMMITS
(small, atomic, well-messaged). Every step has an INTERVIEW LESSON — a story/insight you can
tell in the interview. Keep a `NOTES.md` in the repo: one lesson + one gotcha per day.

Convention: each numbered step = ≥1 commit. Message format: `feat(scope): what & why`.
End of each day: push, write NOTES.md entry, run full test suite.

---

## DAY 1 — Foundation & Data Layer
1. Repo scaffold: FastAPI app factory, settings via pydantic-settings, docker-compose (postgres, redis), ruff+mypy+pytest CI skeleton.
   - Commit: `chore: scaffold fastapi app with compose and ci`
   - Lesson: 12-factor config; why app-factory pattern enables test isolation.
2. Async SQLAlchemy 2.0 setup: async engine, sessionmaker, `get_db` dependency, Base.
   - Commit: `feat(db): async sqlalchemy engine and session dependency`
   - Lesson: async sessions must not be shared across tasks; scoped session vs request-scoped.
3. Domain models: Job (state machine enum), Segment, Asset, UsageRecord. Constraints + FKs + indexes (`(status, created_at)`).
   - Commits: one per model. `feat(models): job state machine` …
   - Lesson: DB-level constraints (check on state transitions, unique idempotency keys) beat app-level validation; index design driven by query patterns you *expect* (reaper scan).
4. Alembic async setup + 3 revisions (init → indexes → usage table). Test upgrade AND downgrade.
   - Lesson: migration hygiene = reversible migrations; autogenerate is a draft, never trust it.
5. Repository layer: thin async repos (JobRepo.get_stale, .claim_next) instead of raw ORM in endpoints.
   - Lesson: repository pattern keeps Celery workers and API sharing one data vocabulary; makes mocking trivial.

**Interview story of the day:** "How I designed indexes from expected access patterns before writing queries."

---

## DAY 2 — Task Fabric: Celery, Backpressure, Idempotency
6. Celery app with per-stage queues (extract/asr/translate/tts/qc/stitch) + routed task base class.
   - Lesson: independent scaling per stage = the whole point; queue depth metrics come free.
7. Idempotency: every job step keyed by `(job_id, stage)`; re-running a completed stage is a no-op. Unique constraint enforces it.
   - Commit: `feat(tasks): idempotent stage execution via unique job-stage key`
   - Lesson: at-least-once delivery means handlers MUST be idempotent; enforce in DB, not memory.
8. Retries with exponential backoff + jitter; permanent vs transient error taxonomy.
   - Lesson: retry storms; why jitter exists (thundering herd).
9. Backpressure: admission control middleware — reject 429 when queue depth > threshold (Redis-based gauge). Load-shed by priority.
   - Lesson: backpressure at ingress beats OOM at workers; 429 + Retry-After is honest API design.
10. Stuck-job reaper: periodic beat task scans stale RUNNING jobs past heartbeat TTL → requeue or DLQ. Heartbeat column updated inside tasks.
    - Lesson: visibility timeout gaps in distributed queues; heartbeats > timeouts.

**Interview story:** "Walk me through what happens when a worker dies mid-translation." (You can answer end-to-end now.)

---

## DAY 3 — Pipeline Spine: Extract → VAD → ASR → Translate
11. Engine abstraction: `Engine` protocol; `SarvamEngine`, `OpenRouterFallbackEngine`, `MockEngine`. Registry + config-selected.
    - Lesson: ports-and-adapters so ML providers are swappable; MockEngine = why your CI needs no API keys.
12. Stage 1–2: ffmpeg audio extraction (16k mono wav) + VAD (silero) segmentation. soundfile I/O utilities.
    - Lesson: sample-rate/format normalization FIRST prevents a class of downstream bugs.
13. Stage 3: faster-whisper local ASR, batched segments, GPU/CPU fallback, word timestamps stored.
    - Lesson: CPU-bound work in async world → run_in_executor / dedicated worker pool; never block the loop.
14. Stage 4: translation via Sarvam w/ OpenRouter fallback + LLM prompt for context window (episode-level glossary).
    - Lesson: fallback orchestration + structured output parsing; provider failure budgeting.
15. End-to-end happy path on one real En→Ta clip; store artifacts (segments json, per-segment audio) in blob layout.
    - Commit: `feat(pipeline): e2e extract-vad-asr-translate on fixture media`
    - Lesson: artifact storage layout (job_id/stage/) = resumability + debuggability.

**Interview story:** "Design decisions in making ML providers pluggable" + live demo of pipeline.

---

## DAY 4 — TTS, QC, Stitch (finish the pipeline)
16. TTS stage: Sarvam TTS engine, per-segment synthesis, tempo matching to original segment duration (time-stretch factor calc).
17. QC stage (JD-critical): SNR check (librosa), duration-ratio score, loudness analysis → guided `ffmpeg loudnorm` normalization, pronunciation spot-check via whisper round-trip similarity score.
    - Lessons: objective QC gates turn "sounds fine" into numbers; round-trip ASR as cheap pronunciation verification.
18. QC feedback loop: failed QC → auto-retry with adjusted parameters (slower TTS rate), max N attempts → flag human review.
    - Lesson: automated remediation hierarchy: parameter tweak → retry → human.
19. Stitch: ffmpeg concat + mux onto original video, subtitle track generation.
20. Full En→Ta dubbed video output. Demo fixture committed (small clip) + golden tests with MockEngine.
    - Lesson: deterministic tests for nondeterministic ML — mock engines + snapshot fixtures.

**Interview story:** "The QC scoring system I designed" — very few candidates have this concrete.

---

## DAY 5 — mozhi-sdk + Real-time Layer
21. Extract cross-cutting code into `mozhi-sdk` package: auth middleware (API key/JWT), rate limiter, metering (UsageRecord writes), OTel tracing helpers that propagate trace context INTO Celery tasks.
    - Lesson: OTel across process boundaries (FastAPI→Celery) via message headers — most people lose traces at the broker.
22. Publish sdk: versioned semver, CHANGELOG.md, backward-compat deprecation policy documented. Installable via pip (private index or PyPI test).
    - Lesson: SDK design = API stability contracts; how you'd deprecate without breaking consumers.
23. WebSocket job tracking: progress events pushed from Celery state callbacks through Redis pub/sub to WS clients.
24. SSE endpoint for live-translation streaming (chunked ASR partials → translations).
25. Refactor main app to consume mozhi-sdk; prove the boundary is clean.
    - Lesson: extracting a library teaches you where your real seams are.

**Interview story:** "Distributed tracing across a message broker" — senior-sounding, rare.

---

## DAY 6 — Kubernetes, Scale, Observability
26. Dockerfiles: multi-role single image (api / worker / beat selected by entrypoint env) — JD names this exactly.
27. Helm chart: deployments per role, HPA/KEDA scaledobject per queue (KEDA on Redis queue length), secrets, ingress.
28. Deploy on local k3s; document cloud deploy path (GKE/EKS notes in README).
29. Observability: Prometheus metrics (per-stage latency, queue depth, DLQ count, QC fail rate), Grafana dashboard JSON committed, structured logging (structlog) with job_id binding.
30. k6 load test: ramp to find the breaking point; record numbers in README (jobs/min per stage, p95 latency, backpressure kick-in point).
    - Lesson: load testing tells you WHERE backpressure should trigger — you set thresholds from evidence.

**Interview story:** "I found our bottleneck with k6 and tuned X" with real numbers. Numbers win interviews.

---

## DAY 5b — Demo Frontend (Studio Console, ~6h) — resume-worthy demo client
Goal: a single-page "Mozhi Studio" console that makes the pipeline VISIBLE. This is the
resume/GIF/interview-demo surface — recruiters click links, they don't run curl.
Stack: React + Vite + TypeScript, Tailwind; served as static files by FastAPI (one deploy unit,
no nginx). State: TanStack Query for REST + native WebSocket hook. No router needed (3 views).

36. Scaffold vite app inside `web/`, proxy `/api` to FastAPI in dev, build output copied into
    `app/static` for prod serving. Commit: `feat(ui): scaffold studio console`
37. Upload view: drag-drop media → POST /jobs → job created toast.
    - Lesson: multipart streaming upload with progress (axios onUploadProgress).
38. Jobs dashboard: list + live per-stage progress via WebSocket
    (`extract ✓ → vad ✓ → asr ⠋ → translate → tts → qc → stitch`) with stage timings.
    - Lesson: reconciling WS state snapshots with REST fallback on reconnect — real-time UX edge cases.
39. Result view: original vs dubbed video side-by-side player, synced play/pause;
    segment table with QC scores (SNR / duration-ratio / loudness), red-flagged failed segments.
40. Live-translation tab: mic or file → SSE partial transcripts + translations appearing live.
41. Polish: dark theme, pipeline diagram header, error states (DLQ'd jobs visible with reason).
42. Record 30–60s demo GIF for README top + resume link. Deploy note: works on k3s ingress.

Resume bullet it enables: "Built Studio console (React/TS/WebSocket) with live per-stage
pipeline visualization and side-by-side dubbing preview for the Mozhi platform."

**Interview story:** "Designing real-time UX over an unreliable WebSocket" (reconnect, snapshot+delta).

---

## DAY 7 — Hardening, Tests, Docs, Application
31. Test suite completion: unit (repos, engines mocked), integration (testcontainers postgres+redis), e2e happy path w/ MockEngine; parallel execution (pytest-xdist); coverage gate in CI.
32. Failure drills: kill worker mid-task → verify reaper recovers; poison message → verify DLQ. Script these (`make chaos-test`) and record output.
33. README: architecture diagram (mermaid), design-decisions section (ORM vs raw SQL, backpressure thresholds, engine abstraction), quickstart, load-test results.
34. Resume update + Sarvam application email: repo link, 3 bullet achievements WITH NUMBERS, note QC system + SDK + tracing as highlights.
35. Buffer: whatever slipped. Then rest.

**Final interview stories bank:** dead-worker recovery drill, backpressure tuning via k6, QC remediation ladder, OTel-across-Celery.

---

## Rules
- Micro commits: ≤ ~200 lines each, green tests between commits where possible.
- NOTES.md: daily entry = 1 lesson + 1 gotcha + 1 thing you'd do differently.
- If behind schedule, cut order (least JD value first): SSE (24) → Grafana dashboards polish (29) → pronunciation QC detail (17c). NEVER cut: reaper/DLQ, backpressure, QC scoring, k6 numbers, SDK extraction.

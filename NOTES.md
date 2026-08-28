# NOTES.md — lessons log

One entry per day: 1 lesson + 1 gotcha + 1 thing I'd do differently.
This file doubles as interview prep — every entry is a war story.

---

## Day 3/4 — Real En→Ta pipeline: Groq translation and stitch

### Progress

- Completed the Day 3 pipeline spine through real media processing: ffmpeg extracts a
  16 kHz mono WAV, Silero VAD finds speech boundaries, and faster-whisper produces an
  English transcript with timestamps.
- Added a Groq translation adapter and `MOZHI_ENGINE_MODE=groq`. Local mode keeps the
  free local ASR/TTS path, but uses Groq for translation when the local translator is
  unavailable.
- Confirmed a real En→Ta translation against Groq with `openai/gpt-oss-20b`; the job
  persisted Tamil translations in `translated.json`.
- Implemented and tested the real stitch stage: concatenate TTS WAV files and mux the
  resulting audio onto the original video as `final.mp4`.
- Added a portable uv setup in the Makefile: store Mozhi's virtual environment on the
  native filesystem and use copy links, avoiding repeated package reinstalls from the
  Windows-mounted project directory.

### Lesson

Provider adapters are more than a convenience. They let the pipeline preserve its
artifact contract (`transcript.json` -> `translated.json`) while a provider, model, or
pricing tier changes underneath. The rest of the system does not need to know whether
translation came from a local model, Groq, OpenRouter, or Sarvam.

### Gotchas and mistakes corrected

1. **A `.env` file is not the same as `os.environ`.** Pydantic Settings reads `.env`,
   while direct `os.environ.get(...)` does not. The Groq adapter now gets the key from
   `Settings`, so API and worker configuration follow one path.
2. **Windows line endings can corrupt HTTP headers.** A Groq key copied into a CRLF
   `.env` file had a trailing carriage return, which HTTPX rejected. API-key validators
   now trim surrounding whitespace before the key is used.
3. **Do not remove existing environment naming conventions casually.** The app already
   used `MOZHI_DATABASE_URL` and `MOZHI_REDIS_URL`. Configuration now accepts both the
   existing `MOZHI_*` runtime variables and unprefixed provider keys such as
   `GROQ_API_KEY`; tests protect that compatibility.
4. **Provider model catalogues change.** The original Groq model identifier was not
   available to this account and returned a model-not-found response. Query the Models
   API and record a verified model ID instead of relying on a remembered name.
5. **Patch mocks where code looks up the dependency.** Stitch tests initially patched
   `app.media`, but `stitch_stage` had already imported those functions. Patching
   `app.stages.stitch_stage` made the test intercept the actual call site.
6. **A Celery restart is required after code or environment changes.** Workers are
   long-running processes. Stop old workers, start one uniquely named worker, and keep
   a single worker process during local debugging to avoid duplicate-node warnings and
   tasks being consumed by an older process.

### What I would do differently

- Add a startup configuration diagnostic that reports only boolean key presence and
  selected engine mode; it would have exposed the `.env` versus `os.environ` mismatch
  immediately without leaking secrets.
- Add a small `make verify-groq` command that calls the Models API and confirms the
  configured model before an end-to-end job starts.
- Use a unique `-n` worker name and `-P solo` for Windows development from day one;
  switch to the normal prefork pool only in Linux/WSL production-like runs.

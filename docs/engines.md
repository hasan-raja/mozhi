# Engine Benchmark — Mozhi Translation Layer

## Strategy: Free-First Ladder

| Mode | ASR | Translate | TTS | Use Case |
|------|-----|-----------|-----|----------|
| `MOZHI_ENGINE_MODE=local` | faster-whisper (CPU) | **Groq: openai/gpt-oss-20b** (primary) → OpenRouter: google/gemma-4-31b-it (fallback) | edge-tts (PallaviNeural) | **Development default** — zero cost, runs locally |
| `MOZHI_ENGINE_MODE=groq` | faster-whisper (CPU) | **Groq: openai/gpt-oss-20b** (forced) | edge-tts (PallaviNeural) | Explicit Groq translation |
| `MOZHI_ENGINE_MODE=sarvam` | Sarvam ASR | Sarvam Translate | Sarvam Bulbul v2 | Demo/Production upgrade (₹1,000 free credits) |
| `MOZHI_ENGINE_MODE=mock` | Deterministic fake | Deterministic fake | Deterministic fake | CI / load tests — zero keys, zero latency |

---

## Translation Engine Comparison (En→Ta)

| Provider / Model | Free Tier | Latency (per 25 seg) | Quality (subjective) | Cost/10k chars | Notes |
|------------------|-----------|---------------------|---------------------|----------------|-------|
| **Groq: openai/gpt-oss-20b** | ✅ Free tier | Measured on a real En→Ta clip | ★★★★☆ Natural dialogue | $0 | **Primary for dev** — confirmed available to this Groq account |
| **OpenRouter: google/gemma-4-31b-it** | ✅ Free tier | ~3–4s (batched) | ★★★★☆ Strong Indic | $0 | Fallback if Groq unavailable |
| **OpenRouter: google/gemini-2.0-flash-001** | ✅ Free tier | ~3–4s (batched) | ★★★★☆ Excellent context | $0 | Previous primary, still available |
| **Sarvam Translate** | ₹1,000 credits | ~2–5s | ★★★★★ Best Indic | ₹20/10k | Demo only — credits cover ~500k chars |
| **IndicTrans2 (local)** | ✅ Free | ~10–30s (CPU) | ★★★☆☆ Good, no context | $0 | Heavy model, GPU preferred; `usable=False` until optimized |

---

## TTS Engine Comparison (Tamil)

| Provider / Voice | Free Tier | Latency (per 25 seg) | Quality | Cost/10k chars | Notes |
|------------------|-----------|---------------------|---------|----------------|-------|
| **edge-tts: ta-IN-PallaviNeural** | ✅ Free | ~5–10s (batched) | ★★★★☆ Natural | $0 | **Primary for dev** — Microsoft neural voices |
| **Piper: tamil** | ✅ Free | ~10–20s (CPU) | ★★★☆☆ Robotic | $0 | Offline, fully local |
| **Sarvam Bulbul v2** | ₹1,000 credits | ~3–5s | ★★★★★ Best Indic | ₹15/10k | Demo only |

---

## ASR Engine Comparison

| Provider / Model | Free Tier | Latency (5 min audio) | Quality (WER) | Cost/min | Notes |
|------------------|-----------|---------------------|--------------|----------|-------|
| **faster-whisper: base.en** | ✅ Free | ~30–60s (CPU) | ~5–8% | $0 | **Primary for dev** — good accuracy/speed balance |
| **Sarvam ASR** | ₹1,000 credits | ~10–20s | ~3–5% | Varies | Demo only |

---

## Decision: Option A — Minimal Changes

**Keep local engines for STT and TTS** — faster-whisper + edge-tts/Piper remain on your machine.

**Swap only the translation layer to Groq:**
- Primary: `openai/gpt-oss-20b` (confirmed available through the Groq Models API)
- Fallback: `google/gemma-4-31b-it` via OpenRouter (if Groq quota exhausted)

**Keep Sarvam only for final demo recording** — ₹1,000 credits cover ~500k chars at ₹20/10k.

---

## How to Run

```bash
# Development (default) — uses Groq for translation
export MOZHI_ENGINE_MODE=local
# Requires: GROQ_API_KEY in .env (free at console.groq.com/keys)

# Explicit Groq mode — forces Groq translation
export MOZHI_ENGINE_MODE=groq

# Demo mode — uses Sarvam APIs
export MOZHI_ENGINE_MODE=sarvam
# Requires: SARVAM_API_KEY in .env

# CI / testing — no keys needed
export MOZHI_ENGINE_MODE=mock
```

---

## Benchmark Notes

- All latency measurements on CPU-only WSL2 (Ubuntu 22.04), 16GB RAM
- Translation measured on 25 segments (~300 words) En→Ta
- TTS measured on 25 segments (~300 words) Tamil synthesis
- Quality scores subjective; based on listening to 3 test clips
- Run `make benchmark` (TODO) to regenerate these numbers
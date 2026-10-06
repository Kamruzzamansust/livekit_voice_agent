# Trip Mate (English) — Bangladesh Trip Voice Agent

An English-only voice assistant that answers questions about three trip plans
(Saint Martin's Island, Khulna & the Sundarbans, Sylhet & Sunamganj) using RAG.
Speech-to-text and text-to-speech are open-source models running on the CPU.

```
mic → Silero VAD → faster-whisper (STT) → FAISS search → Groq LLM (one call) → Piper / Kokoro (TTS) → speaker
                                              (bge-small-en-v1.5, filtered to the trip being discussed)
```

| Part | Model | License | Size | Runs on |
|---|---|---|---|---|
| VAD | Silero VAD | MIT | ~2 MB | CPU |
| STT | faster-whisper `base.en` (int8) | MIT | ~145 MB | CPU |
| TTS | Piper `en_US-lessac-medium` (default) | MIT | 60 MB | CPU |
| TTS (optional) | Kokoro-82M int8 | Apache-2.0 | 86 MB | CPU (slower) |
| Turn detection | LiveKit turn detector `v1-mini` | LiveKit model license | ~108 MB | CPU |
| Embeddings | BAAI/bge-small-en-v1.5 (fastembed ONNX) | MIT | ~67 MB | CPU |
| LLM | Groq (`openai/gpt-oss-120b`) | — | cloud API | — |

| File | Purpose |
|---|---|
| `data/trips/*.txt` | The three English trip plans (the knowledge base) |
| `embeddings.py` | bge-small-en-v1.5 embeddings via fastembed (no PyTorch) |
| `ingest.py` | Chunks + embeds the trip plans into `vector_store/` (FAISS) |
| `trip_agent.py` | Retrieval (`with_trip_context`) + LangChain `create_agent` with Groq |
| `local_speech.py` | LiveKit adapters for faster-whisper STT and Piper / Kokoro TTS |
| `main.py` | LiveKit voice agent entrypoint |
| `config.py` | Settings (overridable from `.env`) |

## Setup

1. `.env` needs `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `GROQ_API_KEY` (use `[PLACEHOLDER]`
   values in any shared example; never commit `.env`).
   Optional: `GROQ_MODEL`, `WHISPER_MODEL` (`tiny.en`/`base.en`/`small.en`), `WHISPER_CPU_THREADS`,
   `TTS_ENGINE` (`piper`/`kokoro`), `KOKORO_VOICE`.
2. Install: `uv sync`
3. Download the TTS voices into `models/`:
   ```bash
   cd models
   B=https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium
   curl -LO $B/en_US-lessac-medium.onnx && curl -LO $B/en_US-lessac-medium.onnx.json
   # Optional Kokoro (nicer voice, slower on CPU):
   K=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
   curl -LO $K/kokoro-v1.0.int8.onnx && curl -LO $K/voices-v1.0.bin
   ```
4. Build the vector store: `uv run python ingest.py` (downloads the embedder on first run;
   re-run after editing any trip file). Whisper weights download on the first agent start.

## Run

```bash
uv run python trip_agent.py "What is the budget for Saint Martin?"  # text-only RAG test
uv run python main.py console   # talk to it with your laptop mic + speaker
uv run python main.py dev       # connect to LiveKit Cloud; talk via the Agents Playground
```

Try: *"When is the best time to visit Tanguar Haor?"*, *"How do I get to Saint Martin from Dhaka?"*,
*"Give me the Sundarbans day-by-day plan."*

## CPU tuning

- Measured on this laptop: Piper makes 3.5 s of speech in ~0.4 s; `base.en` transcribes it in ~1 s;
  STT + TTS together use ~600 MB RAM.
- Slower machine: `WHISPER_MODEL=tiny.en`. Better accuracy on place names: `WHISPER_MODEL=small.en`.
- Kokoro sounds more natural but is several times slower than Piper on CPU.

## Latency

Retrieval runs before the LLM on every turn instead of as a tool call, so each turn is one Groq
request (~0.6–0.9 s to first token) instead of two, and uses about half the tokens. Groq's free tier
allows ~8,000 tokens/minute; if replies sometimes stall, look for `Retrying request` in the logs,
which means Groq rate-limited the call.

## Notes

- Trip-plan prices and rules are approximate (2025–26). Saint Martin access rules change every season.
- `vector_store/index.pkl` is loaded with pickle; only load indexes you built yourself.
- Voice audio goes only to local models; transcribed text (and RAG snippets) is sent to Groq for the LLM.

# Trip Mate (English) — Bangladesh Trip Voice Agent

An English-only voice assistant that answers questions about three trip plans
(Saint Martin's Island, Khulna & the Sundarbans, Sylhet & Sunamganj) using RAG.
Speech runs on LiveKit Inference by default (Deepgram Nova-3 + Inworld TTS-2), or on open-source
CPU models with `SPEECH_MODE=local`.

```
mic → Silero VAD → STT → agent → TTS → speaker

SPEECH_MODE=cloud (default): Deepgram Nova-3 STT + Inworld TTS-2 (voice Ashley) via LiveKit Inference
SPEECH_MODE=local          : faster-whisper STT + Piper / Kokoro TTS on this CPU

AGENT_MODE=rag  (default): FAISS search first → one LLM call                 (trip_agent.py)
AGENT_MODE=deep          : deepagents agent → search_trip_plans tool
                           └─ task tool → itinerary-planner subagent → search_trip_plans   (deep_agent.py)

LLM_BACKEND=ollama (default): Ollama server, e.g. a GPU box on the LAN (OLLAMA_BASE_URL, OLLAMA_MODEL)
LLM_BACKEND=groq           : Groq cloud API
LLM_BACKEND=local          : llama.cpp llama-server on this CPU (Qwen3-1.7B Q4_K_M, thinking off)
```

| Part | Model | License | Size | Runs on |
|---|---|---|---|---|
| VAD | Silero VAD | MIT | ~2 MB | CPU |
| STT | faster-whisper `base.en` (int8) | MIT | ~145 MB | CPU |
| TTS | Piper `en_US-lessac-medium` (default) | MIT | 60 MB | CPU |
| TTS (optional) | Kokoro-82M int8 | Apache-2.0 | 86 MB | CPU (slower) |
| Turn detection | LiveKit turn detector `v1-mini` | LiveKit model license | ~108 MB | CPU |
| Embeddings | BAAI/bge-small-en-v1.5 (fastembed ONNX) | MIT | ~67 MB | CPU |
| LLM (default) | Ollama `gemma3:12b` (rag) / `gemma4:12b` (rag or deep) | Gemma license | 8 GB | GPU server |
| LLM (option) | Groq (`openai/gpt-oss-120b`) | — | cloud API | — |
| LLM (local) | Qwen3-1.7B Q4_K_M via llama.cpp `llama-server` | Apache-2.0 | ~1.1 GB | CPU |

| File | Purpose |
|---|---|
| `data/trips/*.txt` | The three English trip plans (the knowledge base) |
| `embeddings.py` | bge-small-en-v1.5 embeddings via fastembed (no PyTorch) |
| `ingest.py` | Chunks + embeds the trip plans into `vector_store/` (FAISS) |
| `trip_agent.py` | Retrieval (`with_trip_context`), `make_llm()` (Groq or local), rag-mode agent |
| `deep_agent.py` | Deep agent (`create_deep_agent`) with the search tool and an itinerary-planner subagent |
| `llm_server.py` | Starts the local llama.cpp server (localhost only) |
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

5. Optional, for `LLM_BACKEND=local`: build llama.cpp's server and download the model
   (~5–8 min build, ~1.1 GB model):
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
   cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build llama.cpp/build --target llama-server -j 6
   curl -L -o models/Qwen3-1.7B-Q4_K_M.gguf \
     https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf
   ```

## Run

```bash
uv run python trip_agent.py "What is the budget for Saint Martin?"  # text-only RAG test
uv run python main.py console   # talk to it with your laptop mic + speaker
uv run python main.py dev       # connect to LiveKit Cloud; talk via the Agents Playground

# Deep agent and/or local LLM (set in .env or inline)
AGENT_MODE=deep uv run python deep_agent.py "Compare budget trips to Saint Martin and Sylhet"
uv run python llm_server.py                              # terminal 1: local LLM server
LLM_BACKEND=local uv run python main.py console          # terminal 2
LLM_BACKEND=local AGENT_MODE=deep uv run python main.py console
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

## Ollama

Set in `.env`: `LLM_BACKEND=ollama`, `OLLAMA_BASE_URL=http://[OLLAMA_HOST]:11434`, `OLLAMA_MODEL=gemma3:12b`.
`AGENT_MODE=deep` needs a model with tool calling: `gemma3` has none (the agent stops at startup
with a clear error), so use `gemma4:12b`. Thinking is switched off automatically for thinking models.

Measured with the GPU server (model already loaded):

| Setup | First word | Whole answer |
|---|---|---|
| rag + gemma3:12b | ~0.6–0.7 s | ~1–2 s |
| deep + gemma4:12b, simple question | ~1.7 s | ~2.5 s |
| deep + gemma4:12b, trip comparison (subagent) | ~1.0 s (filler) | ~7.5 s |

The first request after a model switch takes 10–25 s while Ollama loads it into GPU memory;
`OLLAMA_KEEP_ALIVE` (default 30m) keeps it loaded between turns. Avoid the `:cloud` models in
Ollama for this project: they run on Ollama's servers, so the conversation would leave the LAN.

## Local LLM and deep agent: what to expect on this laptop

Measured on the i5-10210U (4 cores, throttles to ~2.4 GHz under load, ~1.6 GB free RAM):
llama.cpp processes prompts at ~65 tokens/s and generates ~16–20 tokens/s with 1–1.5B models.

| Setup | First word after you stop talking | Whole answer | Notes |
|---|---|---|---|
| rag + Groq (default) | ~0.6–1.5 s | ~2 s | Accurate |
| deep + Groq | ~1.3–5 s | ~3–8 s | 2–4 LLM calls per turn, accurate |
| rag + local Qwen2.5-1.5B | ~9–14 s | ~10–17 s | Weaker answers |
| deep + local Qwen3-1.7B, simple question | ~13 s ("Let me check the trip plans") | ~29 s | Correct |
| deep + local Qwen3-1.7B, trip comparison | ~1.3 s (filler) | ~85–145 s | Long answer, mixes per-night and per-person prices |

Model notes from testing:
- Qwen2.5-1.5B never called a tool on its own (through llama-cpp-python or llama-server), so it can't drive the deep agent.
- Qwen3-1.7B calls tools and delegates to the subagent correctly; its summaries are the weak point.
- Llama-3.2-1B said swimming in Ratargul is allowed, which is dangerous; removed.

The local options are for learning and experiments, not for real travel advice.

## Notes

- Trip-plan prices and rules are approximate (2025–26). Saint Martin access rules change every season.
- `vector_store/index.pkl` is loaded with pickle; only load indexes you built yourself.
- With `SPEECH_MODE=cloud`, your voice audio and the agent's replies go through LiveKit Cloud to
  Deepgram and Inworld (billed to the LiveKit project in `.env`). With `SPEECH_MODE=local`, audio
  stays on this laptop.
- Voice audio goes only to local models when `SPEECH_MODE=local`; with `LLM_BACKEND=ollama`, transcribed text goes to the
  Ollama server on the LAN (unencrypted HTTP). With `LLM_BACKEND=groq`, transcribed text (and RAG snippets) is
  sent to Groq. With `LLM_BACKEND=local` nothing leaves the machine except LiveKit signalling.
- The local LLM server listens on 127.0.0.1 only and has no authentication; don't expose its port.

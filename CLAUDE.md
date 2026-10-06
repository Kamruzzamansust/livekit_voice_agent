# Second Voice Agent

"Trip Mate" (English only): a LiveKit voice agent that answers Bangladesh trip questions via RAG. See [README.md](README.md) for structure and commands.

- Python via `uv` (`uv run python ...`). Rebuild the FAISS index with `uv run python ingest.py` after editing `data/trips/` or changing `EMBEDDING_MODEL`.
- STT/TTS are local open-source models (faster-whisper, Piper/Kokoro) wrapped as non-streaming LiveKit plugins in `local_speech.py`; the LLM is Groq via a tool-less LangChain `create_agent` graph in `trip_agent.py`. RAG runs before the LLM (`with_trip_context`, called from `SpeechOnlyGraph` in `main.py`) so each turn is a single LLM call. No cloud speech mode.
- Target hardware is a laptop CPU with little free RAM (~1-2 GB): keep models small and `num_idle_processes=1`.

## LiveKit documentation

LiveKit is a fast-evolving project. Always refer to the latest documentation rather than training data.

Use the LiveKit Docs MCP server (`livekit-docs`, provided by the `livekit` plugin) for all lookups. Do not use the `lk docs` CLI. Prefer browsing over search, and `docs_search` over `code_search`.

## Testing

- Practice TDD with the SDK's testing tools (pytest): assert on messages, tool calls, and handoffs.
- Before calling a change done, try it in a live text conversation with `lk agent debugger` (`start`, `say "..."`, `say --logs` for tool tracebacks). Restart the debugger after every code edit.

## Secrets

Keep `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and provider keys in `.env` / `.env.local` (never committed). Use `[PLACEHOLDER]` values in examples.

"""Central settings, loaded from .env. Secrets never have defaults here."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


# LiveKit + Groq credentials (read by the SDKs from the environment too)
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
GROQ_API_KEY = _required("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# gpt-oss models accept low/medium/high (low is fastest); leave empty for other models
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# Knowledge base + vector store
TRIPS_DIR = ROOT / "data" / "trips"
VECTOR_STORE_DIR = ROOT / "vector_store"
MODELS_DIR = ROOT / "models"
# Small English-only embedder (~67 MB ONNX, runs on CPU via fastembed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 3
# Only the last few messages are sent to the LLM each turn (keeps prompts small and fast)
MAX_HISTORY_MESSAGES = 6

# Open-source speech models, run locally on CPU (English only)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")  # tiny.en / base.en / small.en
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")  # piper (fast on CPU) or kokoro (nicer voice)
PIPER_MODEL_PATH = MODELS_DIR / "en_US-lessac-medium.onnx"
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.int8.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

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


LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
MODELS_DIR = ROOT / "models"

# LLM: "ollama" = an Ollama server, e.g. a GPU box on the LAN (OLLAMA_BASE_URL, OLLAMA_MODEL)
#      "groq"   = Groq cloud API (needs GROQ_API_KEY)
#      "local"  = llama.cpp server on this CPU (start it with `uv run python llm_server.py`);
#                 slow: on the i5-10210U a 1.7B model needs 15-30 s per reply
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# gemma3 has no tool calling in Ollama, so AGENT_MODE=deep needs e.g. gemma4:12b or qwen3-vl
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
# Keep the model loaded in GPU memory between turns (Ollama unloads idle models after 5 min)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Qwen3-1.7B calls tools reliably (needed for AGENT_MODE=deep); Qwen2.5-1.5B never did in tests
LOCAL_LLM_PATH = MODELS_DIR / os.getenv("LOCAL_LLM_FILE", "Qwen3-1.7B-Q4_K_M.gguf")
LOCAL_LLM_HOST = "127.0.0.1"  # localhost only: the server has no authentication
LOCAL_LLM_PORT = int(os.getenv("LOCAL_LLM_PORT", "8080"))
LOCAL_LLM_URL = f"http://{LOCAL_LLM_HOST}:{LOCAL_LLM_PORT}/v1"
# Shared secret between llm_server.py and the agent (generate once, keep it in .env)
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "").strip()
LOCAL_LLM_CONTEXT = int(os.getenv("LOCAL_LLM_CONTEXT", "4096"))
LLAMA_SERVER_BIN = ROOT / "llama.cpp" / "build" / "bin" / "llama-server"

# Agent: "rag"  = retrieve first, then one LLM call (fast; trip_agent.py)
#        "deep" = deepagents agent that calls the search tool and a planner subagent itself
#                 (more capable, several LLM calls per turn; deep_agent.py)
AGENT_MODE = os.getenv("AGENT_MODE", "rag")
# Physical cores, not hyper-threads, give the best llama.cpp speed
LOCAL_LLM_THREADS = int(os.getenv("LOCAL_LLM_THREADS", "4"))
# Spoken answers are 2-3 sentences; a hard cap keeps a rambling small model from hogging the CPU
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "160"))

GROQ_API_KEY = _required("GROQ_API_KEY") if LLM_BACKEND == "groq" else ""
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# gpt-oss models accept low/medium/high (low is fastest); leave empty for other models
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# Knowledge base + vector store
TRIPS_DIR = ROOT / "data" / "trips"
VECTOR_STORE_DIR = ROOT / "vector_store"
# Small English-only embedder (~67 MB ONNX, runs on CPU via fastembed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
# 4, not 3: misheard questions ("KNI through plastic" for "Can I throw plastic") rank the
# right chunk lower
TOP_K = int(os.getenv("TOP_K", "4"))
# Only the last few messages are sent to the LLM each turn (keeps prompts small and fast)
MAX_HISTORY_MESSAGES = 6

# Speech: "cloud" = LiveKit Inference (Deepgram Nova-3 STT + Inworld TTS-2), billed to the
#                  LiveKit Cloud project in .env; audio leaves this machine
#         "local" = open-source models on this CPU (faster-whisper + Piper/Kokoro)
SPEECH_MODE = os.getenv("SPEECH_MODE", "cloud")
CLOUD_STT_MODEL = os.getenv("CLOUD_STT_MODEL", "deepgram/nova-3")
CLOUD_STT_LANGUAGE = os.getenv("CLOUD_STT_LANGUAGE", "multi")
CLOUD_TTS_MODEL = os.getenv("CLOUD_TTS_MODEL", "inworld/inworld-tts-2")
CLOUD_TTS_VOICE = os.getenv("CLOUD_TTS_VOICE", "Ashley")
# Seconds of silence before the agent replies: at least min_delay, at most max_delay (the turn
# detector decides in between). 6.0 made short questions like "Hello?" wait 6 s per pause, and a
# noisy mic kept restarting that wait; 3.0 is LiveKit's default.
ENDPOINTING_MIN_DELAY = float(os.getenv("ENDPOINTING_MIN_DELAY", "0.5"))
ENDPOINTING_MAX_DELAY = float(os.getenv("ENDPOINTING_MAX_DELAY", "3.0"))
# Silero voice-activity detection: raise the threshold (up to ~0.8) if noise or the agent's own
# voice from the speakers is detected as user speech
VAD_ACTIVATION_THRESHOLD = float(os.getenv("VAD_ACTIVATION_THRESHOLD", "0.6"))
VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.2"))
# Words the STT must hear before the user can interrupt the agent (0 = any voice activity)
INTERRUPTION_MIN_WORDS = int(os.getenv("INTERRUPTION_MIN_WORDS", "2"))

# Open-source speech models, run locally on CPU (SPEECH_MODE=local)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")  # tiny.en / base.en / small.en
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")  # piper (fast on CPU) or kokoro (nicer voice)
PIPER_MODEL_PATH = MODELS_DIR / "en_US-lessac-medium.onnx"
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.int8.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

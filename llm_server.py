"""Start the local LLM: llama.cpp's OpenAI-compatible llama-server, CPU only.

Run it in its own terminal and leave it running:
    uv run python llm_server.py
The voice agent (LLM_BACKEND=local) then talks to it at config.LOCAL_LLM_URL.

We use the native llama-server (built into llama.cpp/build/bin, see README) rather than
llama-cpp-python's server: with --jinja it formats and parses tool calls using the model's
own chat template, which the deep agent needs. llama-cpp-python's generic
chatml-function-calling format produced no tool calls at all with Qwen2.5 in tests.
"""

import os

import config

if not config.LOCAL_LLM_PATH.exists():
    raise SystemExit(f"Model not found: {config.LOCAL_LLM_PATH}. See README.md to download it.")
if not config.LLAMA_SERVER_BIN.exists():
    raise SystemExit(f"llama-server not found at {config.LLAMA_SERVER_BIN}. See README.md to build it.")
if not config.LOCAL_LLM_API_KEY:
    raise SystemExit(
        "Set LOCAL_LLM_API_KEY in .env first, for example:\n"
        '  echo "LOCAL_LLM_API_KEY=$(openssl rand -hex 32)" >> .env'
    )

args = [
    str(config.LLAMA_SERVER_BIN),
    "--model", str(config.LOCAL_LLM_PATH),
    "--host", config.LOCAL_LLM_HOST,
    "--port", str(config.LOCAL_LLM_PORT),
    "--ctx-size", str(config.LOCAL_LLM_CONTEXT),
    "--threads", str(config.LOCAL_LLM_THREADS),
    "--n-gpu-layers", "0",
    "--parallel", "1",  # one conversation at a time: a single KV cache keeps RAM low
    "--jinja",  # use the GGUF's chat template, including its tool-call format
    # Qwen3 "thinks" before answering by default: far too slow on CPU and it would be spoken.
    # Models without a thinking mode (Qwen2.5, Llama) ignore it.
    "--reasoning", "off",
    "--no-ui",  # the agent only needs the API, not the built-in chat page
]
print("Starting:", " ".join(args), flush=True)
# Even on 127.0.0.1, any web page open in a browser can send requests to this port, so the
# server requires an API key. It goes in through the environment, not argv, so it doesn't
# show up in `ps`.
os.execve(args[0], args, {**os.environ, "LLAMA_API_KEY": config.LOCAL_LLM_API_KEY})

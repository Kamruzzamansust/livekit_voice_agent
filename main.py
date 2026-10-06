"""Trip Mate voice agent (English, open-source speech on CPU).

Pipeline: mic -> Silero VAD -> faster-whisper STT -> LangChain agent (Groq + FAISS RAG)
          -> Piper/Kokoro TTS -> speaker

Run in the terminal (uses your mic/speaker):  uv run python main.py console
Run as a worker for the LiveKit playground:   uv run python main.py dev
"""

import asyncio

from langchain_core.messages import SystemMessage, ToolMessage
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobProcess,
    TurnHandlingOptions,
    inference,
)
from livekit.plugins import langchain as lk_langchain
from livekit.plugins import silero

import config
from local_speech import KokoroTTS, PiperTTS, WhisperSTT
from trip_agent import trip_agent, with_trip_context

GREETING = (
    "Hi, I'm Trip Mate. I can help you plan a trip to Saint Martin's Island, "
    "the Sundarbans from Khulna, or Sylhet and Sunamganj. Where would you like to go?"
)


class SpeechOnlyGraph:
    """Wraps the agent graph so that:
    - only the last few messages are sent to the LLM (smaller prompts, faster and cheaper)
    - the trip notes for the latest question are retrieved and attached before the LLM call
    - tool outputs (raw RAG chunks) are never spoken
    """

    def __init__(self, graph) -> None:
        self._graph = graph

    def astream(self, state, *args, **kwargs):
        messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        messages = messages[-config.MAX_HISTORY_MESSAGES :]

        async def _filtered():
            # Embedding + FAISS search is blocking CPU work; keep it off the event loop
            enriched = await asyncio.to_thread(with_trip_context, messages)
            inner = self._graph.astream({**state, "messages": enriched}, *args, **kwargs)
            async for item in inner:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage):
                    continue
                yield item

        return _filtered()


class TripMateAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            # The system prompt lives in trip_agent.py (passed to create_agent); these
            # instructions are dropped by SpeechOnlyGraph so they aren't sent twice
            instructions="You are Trip Mate, a voice travel assistant for Bangladesh trips.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(trip_agent)),
        )


def prewarm(proc: JobProcess) -> None:
    # Load the models once per worker process, not once per call
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["stt"] = WhisperSTT(
        model=config.WHISPER_MODEL, cpu_threads=config.WHISPER_CPU_THREADS
    )
    if config.TTS_ENGINE == "kokoro":
        proc.userdata["tts"] = KokoroTTS(
            str(config.KOKORO_MODEL_PATH), str(config.KOKORO_VOICES_PATH), voice=config.KOKORO_VOICE
        )
    else:
        proc.userdata["tts"] = PiperTTS(str(config.PIPER_MODEL_PATH))


server = AgentServer(
    setup_fnc=prewarm,
    # Each process holds its own copy of the speech models, so keep just one warm on CPU
    num_idle_processes=1,
    # First run downloads the Whisper weights, so allow extra time to load
    initialize_process_timeout=180.0,
    job_memory_warn_mb=1500,
)


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=ctx.proc.userdata["stt"],
        tts=ctx.proc.userdata["tts"],
        turn_handling=TurnHandlingOptions(
            # v1-mini is LiveKit's small local end-of-turn model: it runs on this CPU, so
            # turn detection doesn't depend on (or bill to) LiveKit Cloud inference
            turn_detection=inference.TurnDetector(version="v1-mini"),
            # Preemptive generation starts LLM calls before the user finishes speaking and
            # throws them away if they keep talking; off to save CPU and LLM rate limits
            preemptive_generation={"enabled": False},
        ),
    )

    await session.start(agent=TripMateAgent(), room=ctx.room)

    await session.say(GREETING)


if __name__ == "__main__":
    agents.cli.run_app(server)

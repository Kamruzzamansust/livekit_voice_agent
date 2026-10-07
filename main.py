"""Trip Mate voice agent (English, open-source speech on CPU).

Pipeline: mic -> Silero VAD -> faster-whisper STT -> LangChain agent (Groq + FAISS RAG)
          -> Piper/Kokoro TTS -> speaker

Run in the terminal (uses your mic/speaker):  uv run python main.py console
Run as a worker for the LiveKit playground:   uv run python main.py dev
"""

import asyncio

from langchain_core.messages import AIMessageChunk, SystemMessage, ToolMessage
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobProcess,
    TurnHandlingOptions,
    inference,
    tokenize,
)
from livekit.plugins import langchain as lk_langchain
from livekit.plugins import silero

import config
from trip_agent import trip_agent, with_trip_context

PLACE_KEYTERMS = [
    "Sylhet", "Sunamganj", "Tanguar Haor", "Tahirpur", "Jaflong", "Ratargul", "Lalakhal",
    "Bisnakandi", "Niladri", "Khulna", "Sundarbans", "Mongla", "Bagerhat", "Saint Martin",
    "Cox's Bazar", "Teknaf", "taka",
]

GREETING = (
    "Hi, I'm Trip Mate. I can help you plan a trip to Saint Martin's Island, "
    "the Sundarbans from Khulna, or Sylhet and Sunamganj. Where would you like to go?"
)


class SpeechOnlyGraph:
    """Wraps the agent graph so that:
    - only the last few messages are sent to the LLM (smaller prompts, faster and cheaper)
    - in rag mode, the trip notes for the latest question are retrieved and attached first
    - tool outputs (raw RAG chunks, subagent reports) are never spoken
    """

    def __init__(self, graph, retrieve_first: bool) -> None:
        self._graph = graph
        self._retrieve_first = retrieve_first

    def astream(self, state, *args, **kwargs):
        messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        messages = messages[-config.MAX_HISTORY_MESSAGES :]

        async def _filtered():
            if self._retrieve_first:
                # Embedding + FAISS search is blocking CPU work; keep it off the event loop
                messages[:] = await asyncio.to_thread(with_trip_context, messages)
            inner = self._graph.astream({**state, "messages": messages}, *args, **kwargs)
            announced = False
            async for item in inner:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage):
                    continue
                # A deep agent turn can take a while (tool calls, subagent); say something as
                # soon as the first tool call starts so the user isn't left in silence
                if not announced and isinstance(item, tuple) and getattr(item[0], "tool_call_chunks", None):
                    announced = True
                    yield AIMessageChunk(content="Let me check the trip plans. "), item[1]
                yield item

        return _filtered()


class TripMateAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            # The system prompt lives in trip_agent.py (passed to create_agent); these
            # instructions are dropped by SpeechOnlyGraph so they aren't sent twice
            instructions="You are Trip Mate, a voice travel assistant for Bangladesh trips.",
            llm=lk_langchain.LLMAdapter(graph=_build_graph()),
        )


def _build_graph() -> SpeechOnlyGraph:
    if config.AGENT_MODE == "deep":
        from deep_agent import trip_deep_agent

        return SpeechOnlyGraph(trip_deep_agent, retrieve_first=False)
    return SpeechOnlyGraph(trip_agent, retrieve_first=True)


def prewarm(proc: JobProcess) -> None:
    # Load the models once per worker process, not once per call
    proc.userdata["vad"] = silero.VAD.load(
        # Higher than the 0.5 default and a longer minimum: steady background noise or the
        # speaker leaking into the mic otherwise counts as "speech" and holds the turn open
        activation_threshold=config.VAD_ACTIVATION_THRESHOLD,
        min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
    )
    # The transcript synchronizer builds a hyphenation table on first use; doing it here
    # avoids a ~140 ms event-loop stall during the first spoken reply
    tokenize.basic.hyphenate_word("warmup")
    if config.SPEECH_MODE == "cloud":
        return

    from local_speech import KokoroTTS, PiperTTS, WhisperSTT

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


def _speech_models(ctx: agents.JobContext):
    """(stt, tts, turn detector) for SPEECH_MODE."""
    if config.SPEECH_MODE == "cloud":
        # LiveKit Inference: billed to the LiveKit Cloud project in .env, no extra API keys
        stt = inference.STT(
            model=config.CLOUD_STT_MODEL,
            language=config.CLOUD_STT_LANGUAGE,
            # Deepgram keyterm prompting: bias recognition toward the trip place names
            extra_kwargs={"keyterm": PLACE_KEYTERMS},
        )
        tts = inference.TTS(model=config.CLOUD_TTS_MODEL, voice=config.CLOUD_TTS_VOICE)
        return stt, tts, inference.TurnDetector()
    # v1-mini is LiveKit's small local end-of-turn model: it runs on this CPU, so turn
    # detection doesn't depend on (or bill to) LiveKit Cloud inference
    return ctx.proc.userdata["stt"], ctx.proc.userdata["tts"], inference.TurnDetector(version="v1-mini")


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    stt, tts, turn_detector = _speech_models(ctx)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        tts=tts,
        turn_handling=TurnHandlingOptions(
            turn_detection=turn_detector,
            endpointing={
                "mode": "fixed",
                "min_delay": config.ENDPOINTING_MIN_DELAY,
                "max_delay": config.ENDPOINTING_MAX_DELAY,
            },
            # Only real words (from STT) may cut the agent off: background noise or the agent's
            # own voice leaking from the speakers made VAD "hear" the user and drop replies
            interruption={"min_words": config.INTERRUPTION_MIN_WORDS},
            # Preemptive generation starts LLM calls before the user finishes speaking and
            # throws them away if they keep talking; off to save CPU and LLM rate limits
            preemptive_generation={"enabled": False},
        ),
    )

    await session.start(agent=TripMateAgent(), room=ctx.room)

    await session.say(GREETING)


if __name__ == "__main__":
    agents.cli.run_app(server)

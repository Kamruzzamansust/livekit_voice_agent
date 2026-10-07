"""LiveKit adapters for open-source English speech models that run locally on CPU.

- WhisperSTT: faster-whisper (OpenAI Whisper *.en weights via CTranslate2, int8)
- PiperTTS:   Piper (VITS, ONNX) - fast enough for real time on a laptop CPU (default)
- KokoroTTS:  Kokoro-82M via ONNX - more natural voice, but slower on CPU

All are non-streaming; AgentSession wraps them automatically:
the STT with a VAD-driven StreamAdapter, the TTS with a sentence-by-sentence StreamAdapter.
"""

import asyncio

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import APIConnectOptions, stt, tts, utils
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

WHISPER_SAMPLE_RATE = 16000
KOKORO_SAMPLE_RATE = 24000

# Whisper doesn't know Bangladeshi place names; priming it with them fixes most mishearings
PLACE_NAMES_PROMPT = (
    "Trip to Saint Martin, Cox's Bazar, Teknaf, Chhera Dwip, Khulna, Sundarbans, Bagerhat, "
    "Kotka, Karamjal, Mongla, Sylhet, Sunamganj, Tanguar Haor, Jaflong, Ratargul, Lalakhal, "
    "Bisnakandi, Niladri Lake, Tahirpur, houseboat, taka. What are the must-try foods?"
)


# Biases decoding toward these words (initial_prompt alone still let "Sylhet" become "select")
HOTWORDS = "Sylhet Sunamganj Sundarbans Tanguar Haor Khulna Ratargul Jaflong Tahirpur Saint Martin"


class WhisperSTT(stt.STT):
    def __init__(self, model: str = "base.en", cpu_threads: int = 4) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._model_name = model
        self._model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=cpu_threads)

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "faster-whisper"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        audio = _to_whisper_input(frame)
        text = await asyncio.to_thread(self._transcribe, audio)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LanguageCode("en"), text=text)],
        )

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,  # Silero VAD already trimmed the audio
            initial_prompt=PLACE_NAMES_PROMPT,
            hotwords=HOTWORDS,
            # Short single-utterance clips: skip timestamp decoding and cross-window context
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


def _to_whisper_input(frame: rtc.AudioFrame) -> np.ndarray:
    """int16 PCM at any rate/channels -> mono float32 at 16 kHz."""
    if frame.sample_rate != WHISPER_SAMPLE_RATE:
        resampler = rtc.AudioResampler(
            frame.sample_rate, WHISPER_SAMPLE_RATE, num_channels=frame.num_channels
        )
        frame = rtc.combine_audio_frames(resampler.push(frame) + resampler.flush())
    pcm = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    return pcm.mean(axis=1).astype(np.float32) / 32768.0


class PiperTTS(tts.TTS):
    def __init__(self, model_path: str) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(model_path)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._voice.config.sample_rate,
            num_channels=1,
        )

    @property
    def model(self) -> str:
        return "piper"

    @property
    def provider(self) -> str:
        return "piper-tts"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _LocalTTSStream(tts=self, input_text=text, conn_options=conn_options)

    def generate(self, text: str) -> bytes:
        return b"".join(chunk.audio_int16_bytes for chunk in self._voice.synthesize(text))


class KokoroTTS(tts.TTS):
    def __init__(self, model_path: str, voices_path: str, voice: str = "af_heart") -> None:
        from kokoro_onnx import Kokoro

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=KOKORO_SAMPLE_RATE,
            num_channels=1,
        )
        self._voice = voice
        self._kokoro = Kokoro(model_path, voices_path)

    @property
    def model(self) -> str:
        return "kokoro-82m"

    @property
    def provider(self) -> str:
        return "kokoro-onnx"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _LocalTTSStream(tts=self, input_text=text, conn_options=conn_options)

    def generate(self, text: str) -> bytes:
        samples, _ = self._kokoro.create(text, voice=self._voice, lang="en-us")
        return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class _LocalTTSStream(tts.ChunkedStream):
    """Runs a local model's blocking generate() off the event loop and emits 16-bit PCM."""

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        local_tts: PiperTTS | KokoroTTS = self._tts  # type: ignore[assignment]
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=local_tts.sample_rate,
            num_channels=local_tts.num_channels,
            mime_type="audio/pcm",
        )
        if self._input_text.strip():
            output_emitter.push(await asyncio.to_thread(local_tts.generate, self._input_text))
        output_emitter.flush()

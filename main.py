import os
import io
import re
import time
import wave
import logging
from queue import Queue, Empty
from threading import Lock, Event, Thread

import numpy as np
import sounddevice as sd
import piper
from faster_whisper import WhisperModel
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import platform

from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent

def rel_path(env_var, default_rel):
    val = os.getenv(env_var)
    if val:
        return Path(val).expanduser()
    return (BASE_DIR / default_rel)

PIPER_MODEL_PATH = rel_path("PIPER_MODEL_PATH", "voice/en_US-libritts-high.onnx")
PIPER_CONFIG_PATH = rel_path("PIPER_CONFIG_PATH", "voice/en_US-libritts-high.onnx.json")
LLM_MODEL_PATH = rel_path("LLM_MODEL_PATH", "model/gpt-oss-20b-mxfp4.gguf")

class TTSCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.buffer = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.buffer += token
        if re.search(r"""[.!?](?:["')\]]+)?\s*$""", self.buffer):
            text = self.buffer.strip()
            if text:
                stop_speaking()
                speak(text, blocking=False)
            self.buffer = ""

    def on_llm_end(self, *args, **kwargs):
        # flush any remaining text
        if self.buffer.strip():
            stop_speaking()
            speak(self.buffer.strip(), blocking=False)
            self.buffer = ""
        # signal that LLM is fully done generating
        _tts_done_evt.set()


_tts = None
_whisper = None
_lock = Lock()


# LLM configuration via environment variables with sensible defaults
llm = ChatLlamaCpp(
    model_path=os.getenv("LLM_MODEL_PATH", "model/gpt-oss-20b-mxfp4.gguf"),
    n_ctx=int(os.getenv("LLM_N_CTX", "4096")),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
    n_threads=int(os.getenv("LLM_N_THREADS", str(os.cpu_count() or 8))),
    streaming=True,
    callbacks=[TTSCallbackHandler(), StreamingStdOutCallbackHandler()],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are SAMURAI, an intelligent AI assistant. Your goal is to be helpful, and informative. You can respond in natural, human-like language and use tools when needed to answer questions more accurately. Always explain your reasoning simply when appropriate, and keep your responses conversational and concise.",
        ),
        ("human", "{input}"),
    ]
)

WAKE_WORD = "SAMURAI"
CONVERSATION_TIMEOUT = 20

def choose_samplerate(desired=16000):
    try:
        sd.check_input_settings(samplerate=desired, channels=1, dtype="float32")
        return desired
    except Exception:
        default = int(sd.query_devices(kind="input")["default_samplerate"])
        try:
            sd.check_input_settings(samplerate=default, channels=1, dtype="float32")
            return default
        except Exception:
            # final fallback – pick the first with mono support
            for dev in sd.query_devices():
                try:
                    sd.check_input_settings(device=dev["name"], samplerate=dev["default_samplerate"], channels=1, dtype="float32")
                    return int(dev["default_samplerate"])
                except Exception:
                    continue
            raise RuntimeError("No compatible input device found.")

def maybe_resample(audio, from_sr, to_sr=16000):
    if from_sr == to_sr:
        return audio
    # Minimal resampling without extra deps (ok for speech, upgrade to soxr for HQ)
    import math
    ratio = to_sr / from_sr
    x = np.arange(int(len(audio) * ratio)) / ratio
    idx = np.floor(x).astype(int)
    idx = np.clip(idx, 0, len(audio) - 1)
    return audio[idx]

def get_tts():
    global _tts
    if _tts is None:
        with _lock:
            if _tts is None:
                model = os.getenv("PIPER_MODEL_PATH", "voice/en_US-libritts-high.onnx")
                cfg = os.getenv(
                    "PIPER_CONFIG_PATH",
                    "voice/en_US-libritts-high.onnx.json",
                )
                if not (os.path.isfile(model) and os.path.isfile(cfg)):
                    raise FileNotFoundError("Missing Piper model/config")
                _tts = piper.PiperVoice.load(model, cfg)
                # Validate output settings without altering global defaults
                sd.check_output_settings(
                    samplerate=_tts.config.sample_rate, channels=1, dtype="int16"
                )
    return _tts


def get_whisper():
    global _whisper
    if _whisper is None:
        with _lock:
            if _whisper is None:
                _whisper = WhisperModel(
                    os.getenv("WHISPER_MODEL", "base"),
                    device=os.getenv("WHISPER_DEVICE", "cpu"),
                    compute_type=os.getenv("WHISPER_COMPUTE", "int8"),
                )
    return _whisper


_stop_evt = Event()
_play_queue = Queue()
_play_thread = None
_tts_done_evt = Event()


def _playback_worker():
    while True:
        buffer_bytes, sr, volume = _play_queue.get()
        try:
            _stop_evt.clear()
            with io.BytesIO(buffer_bytes) as buffer:
                with (
                    wave.open(buffer, "rb") as wf,
                    sd.OutputStream(
                        samplerate=sr,
                        channels=wf.getnchannels(),
                        dtype="int16",
                        blocksize=4096,
                    ) as stream,
                ):
                    while not _stop_evt.is_set():
                        chunk = wf.readframes(4096)
                        if not chunk:
                            break
                        arr = np.frombuffer(chunk, dtype=np.int16)
                        if volume != 1.0:
                            arr = np.clip(
                                arr.astype(np.int32) * volume, -32768, 32767
                            ).astype(np.int16)
                        stream.write(arr)
        except Exception:
            logging.exception("Playback failed")
        finally:
            _play_queue.task_done()  # signal the queue that this chunk is finished
            _stop_evt.clear()


def _ensure_play_worker():
    global _play_thread
    if _play_thread is None or not _play_thread.is_alive():
        _play_thread = Thread(target=_playback_worker, daemon=True)
        _play_thread.start()


def stop_speaking():
    _stop_evt.set()
    # Drain any queued audio
    try:
        while True:
            _play_queue.get_nowait()
    except Empty:
        pass


def wait_until_done_speaking():
    while not _play_queue.empty() or (
        _play_thread and _play_thread.is_alive() and not _stop_evt.is_set()
    ):
        time.sleep(0.05)


def speak(text: str, blocking: bool = True, volume: float = 1.0):
    if not isinstance(text, str) or not text.strip():
        return
    volume = float(np.clip(volume, 0.0, 2.0))
    MAX_CHARS = 2000
    text = text[:MAX_CHARS]

    try:
        tts = get_tts()
        sr = tts.config.sample_rate
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            tts.synthesize_wav(text, wf)

        data = buffer.getvalue()
        _ensure_play_worker()

        if blocking:
            _stop_evt.clear()
            try:
                with io.BytesIO(data) as b:
                    with (
                        wave.open(b, "rb") as wf,
                        sd.OutputStream(
                            samplerate=sr,
                            channels=wf.getnchannels(),
                            dtype="int16",
                            blocksize=4096,
                        ) as stream,
                    ):
                        while not _stop_evt.is_set():
                            chunk = wf.readframes(4096)
                            if not chunk:
                                break
                            arr = np.frombuffer(chunk, dtype=np.int16)
                            if volume != 1.0:
                                arr = (
                                    np.clip(
                                        arr.astype(np.int32) * volume, -32768, 32767
                                    )
                                ).astype(np.int16)
                            stream.write(arr)
            except (sd.PortAudioError, OSError) as e:
                logging.exception("Audio playback failed")
                raise RuntimeError(f"Audio playback failed: {e}") from e
            finally:
                _stop_evt.clear()
        else:
            _play_queue.put((data, sr, volume))
    except Exception as e:
        raise RuntimeError(f"TTS synthesis failed: {e}") from e


def listen():
    target_sr = 16000
    in_sr = choose_samplerate(target_sr)
    frames = int(5 * in_sr)  # 5 seconds
    buf = sd.rec(frames, samplerate=in_sr, channels=1, dtype="float32")
    sd.wait()
    audio = buf.reshape(-1)
    audio_16k = maybe_resample(audio, from_sr=in_sr, to_sr=target_sr)

    segments, _ = get_whisper().transcribe(
        audio_16k,
        temperature=0.0,
        beam_size=1,
        vad_filter=True,
        language="en",
    )
    return " ".join(s.text for s in segments)


def main():
    if not (PIPER_MODEL_PATH.is_file() and PIPER_CONFIG_PATH.is_file()):
        raise FileNotFoundError(f"Missing Piper files: {PIPER_MODEL_PATH} / {PIPER_CONFIG_PATH}")
    
    try:
        sd.check_input_settings(samplerate=choose_samplerate(), channels=1, dtype="float32")
    except Exception as e:
        logging.error("Input audio device not compatible. Consider selecting a different microphone or changing Windows sound settings (sample rate / exclusive mode).")
        raise

    if any(ord(c) > 127 for c in str(BASE_DIR)):
        logging.warning("Project path contains non-ASCII characters. If native libs fail to load models, move models to an ASCII-only directory and set paths via .env.")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    conversation_mode = False
    announced_ready = False
    last_interaction_time = None
    while True:
        try:
            if not conversation_mode:
                if not announced_ready:
                    speak("SAMURAI online and waiting.")
                    announced_ready = True
                logging.info("Waiting for wake word.")
                transcript = listen().strip()
                if WAKE_WORD.casefold() in transcript.casefold():
                    logging.info(f"trigger by {transcript}")
                    speak("Yes, Sir, I'm here. What can I help you with?")
                    conversation_mode = True
                    last_interaction_time = time.time()
                    continue
            else:
                if (
                    last_interaction_time
                    and time.time() - last_interaction_time > CONVERSATION_TIMEOUT
                ):
                    logging.info("⌛ Timeout: Returning to wake word mode.")
                    conversation_mode = False
                    announced_ready = False
                    continue

                logging.info("Listening for user input.")
                command = listen().strip()
                logging.info(f"User said: {command}")

                if not command:
                    continue

                messages = prompt.format_messages(input=command, agent_scratchpad="")
                # Stop any ongoing speech before starting a new model response
                stop_speaking()
                _tts_done_evt.clear()
                response = llm.invoke(messages)
                _tts_done_evt.wait()
                _play_queue.join()
                _ = getattr(response, "content", str(response))
                # TTS is streamed by callbacks; avoid double-speaking the final content

                last_interaction_time = time.time()

        except KeyboardInterrupt:
            logging.info("Shutting down...")
            stop_speaking()
            break
        except Exception as e:
            logging.error(f"Error during recognition or tool call: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()

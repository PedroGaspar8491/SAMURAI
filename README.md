# SAMURAI Voice Assistant

Voice-first assistant that:

- Listens for a wake word (“SAMURAI”)
- Captures speech via microphone
- Transcribes with Faster-Whisper
- Streams responses from an LLM (llama.cpp via LangChain)
- Speaks responses with Piper TTS in real time

All main logic is in `main.py`.

## Features

- Wake word: `SAMURAI`
- 5-second listening windows per turn
- Whisper ASR via `faster-whisper`
- LLM via `llama.cpp` (LangChain `ChatLlamaCpp`)
- TTS via `piper-tts`, streamed at sentence boundaries
- Playback powered by `sounddevice` (PortAudio)

## Requirements

- Python 3.9–3.12 (3.10+ recommended)
- PortAudio (required by `sounddevice`)
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt-get install -y libportaudio2`
  - Windows: Prebuilt wheels usually include PortAudio; if not, install via Chocolatey or use official binaries.
- A local llama.cpp GGUF model file
- Piper voice `.onnx` and `.onnx.json` files
- Microphone and speakers

## Installation

1. Create and activate a virtual environment

- macOS/Linux:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
- Windows (PowerShell):
  - `py -m venv .venv`
  - `.venv\Scripts\Activate.ps1`

2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Optional GPU acceleration:

- macOS (Apple Silicon, Metal):

  Reinstall llama.cpp with Metal:

  ```bash
  CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
  ```

- NVIDIA CUDA (for Faster-Whisper):
  Ensure CUDA toolkit/driver is installed and set:
  ```bash
  export WHISPER_DEVICE=cuda
  ```

## Models

### LLM (GGUF for llama.cpp)

- Put your model at: model/your-model.gguf
- Example: Qwen 2.5 1.5B Instruct GGUF (see Hugging Face)
- Configure path via LLM_MODEL_PATH (see Environment below)

### Piper TTS voice

- Download voices from: https://github.com/rhasspy/piper
- Example voice (English): en_US-lessac-high.onnx and en_US-lessac-high.onnx.json
- Place under: voice/
- Configure via PIPER_MODEL_PATH and PIPER_CONFIG_PATH

### Environment

You can customize behavior using environment variables (defaults shown):

- LLM

  - LLM_MODEL_PATH = model/qwen2.5-1.5b-instruct.gguf
  - LLM_N_CTX = 4096
  - LLM_TEMPERATURE = 0.7
  - LLM_MAX_TOKENS = 512
  - LLM_N_THREADS = number of CPU cores

* Whisper (ASR)

  - WHISPER_MODEL = base (or tiny, small, medium, large-v3, etc.)
  - WHISPER_DEVICE = cpu (use cuda if NVIDIA GPU is available)
  - WHISPER_COMPUTE = int8 (alternatives: float16, float32 depending on device)

* Piper (TTS)

  - PIPER_MODEL_PATH = voice/en_US-lessac-high.onnx
  - PIPER_CONFIG_PATH = voice/en_US-lessac-high.onnx.json

* Other tunables (edit main.py to change):

  - WAKE_WORD = SAMURAI
  - CONVERSATION_TIMEOUT = 20 (seconds)
    You can export variables in your shell or create a local .env (note .env is git-ignored).

## Run

1. Ensure your microphone is connected and permitted for terminal/console apps.
2. Ensure the LLM GGUF and Piper voice files are in place.
3. Start:

```bash
python main.py
```

You should hear: “SAMURAI online and waiting.” Then say “SAMURAI” followed by your request.

## Troubleshooting

- PortAudioError / Audio device issues
  - Install PortAudio (see Requirements)
  - Select a default input/output device in your OS sound settings
  - On Linux, check microphone permissions and ALSA/PulseAudio configs
- TTS synthesis failed: Missing Piper model/config
  - Verify PIPER_MODEL_PATH and PIPER_CONFIG_PATH point to existing files
- LLM loading errors (llama.cpp)
  - Ensure the .gguf file exists at LLM_MODEL_PATH
  - If building from source, ensure CMake and a C++ toolchain are available
- High CPU usage
  - Lower LLM_N_THREADS
  - Use a smaller GGUF model
- Whisper performance/accuracy
  - Choose a larger WHISPER_MODEL for better accuracy (requires more compute)
  - Use WHISPER_DEVICE=cuda if you have an NVIDIA GPU

## Notes

- Responses are streamed to TTS at sentence boundaries to reduce latency.
- The assistant returns to wake-word mode after 20s of inactivity.

```text
# Core runtime
numpy>=1.23,<3.0

# Audio I/O
sounddevice>=0.4.6,<0.5

# TTS
piper-tts>=1.2,<2.0

# ASR
faster-whisper>=1.0,<2.0

# LLM + LangChain
langchain>=0.2.12,<0.3
langchain-core>=0.2.12,<0.3
langchain-community>=0.2.12,<0.3
llama-cpp-python>=0.2.90,<0.3
```

# Windows PowerShell

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install python-dotenv numpy sounddevice piper-tts faster-whisper langchain-core langchain-community

# llama-cpp-python CPU wheel:
pip install llama-cpp-python
```

# Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install python-dotenv numpy sounddevice piper-tts faster-whisper langchain-core langchain-community
pip install llama-cpp-python
```

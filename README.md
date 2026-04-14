# 🎙️ Voice-Controlled Local AI Agent

A fully offline voice agent that listens to spoken commands, classifies intent using a local LLM, and executes actions — file creation, code writing, text summarization — without any cloud dependency.

```
🎤 Voice ─→ 🔤 Whisper STT ─→ 🧠 Local LLM ─→ ⚡ Action ─→ 📄 Result
```

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone and Install Python Dependencies](#1-clone-and-install-python-dependencies)
  - [2. Install System Dependencies](#2-install-system-dependencies)
  - [3. Install and Configure Ollama](#3-install-and-configure-ollama)
  - [4. Configure the Secure Folder](#4-configure-the-secure-folder)
- [Usage](#usage)
  - [Command-Line Mode](#command-line-mode)
  - [Streamlit Web UI](#streamlit-web-ui)
- [Hardware Notes & CPU Workarounds](#hardware-notes--cpu-workarounds)
- [Supported Intents](#supported-intents)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Fully local** — no API keys, no cloud services, no data leaves your machine
- **Voice input** — speak naturally; Whisper handles the transcription
- **Intent classification** — a local LLM parses your command into structured JSON
- **Action execution** — files are created, code is written, text is summarized automatically
- **Streamlit dashboard** — a four-stage visual pipeline showing transcript → intent → action → result
- **Dual input modes** — CLI (press-Enter-to-record) and web UI (in-browser mic, file upload, or text)
- **Backend-agnostic** — swap between Ollama and LM Studio with a single config change

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                             │
│                                                              │
│   CLI (sounddevice)  ──┐                                     │
│   Browser mic (WebM) ──┼──→  Audio Normalization  ──┐        │
│   File upload        ──┘     (pydub / ffmpeg)       │        │
│   Text input         ───────────────────────────────┤        │
│                                                     ▼        │
│                     ┌───────────────────┐                    │
│                     │  Whisper STT      │                    │
│                     │  (whisper-base)   │                    │
│                     │  16 kHz mono WAV  │                    │
│                     └────────┬──────────┘                    │
│                              ▼                               │
│                     ┌───────────────────┐                    │
│                     │  Intent Classifier│                    │
│                     │  (Ollama / LM     │                    │
│                     │   Studio LLM)     │                    │
│                     │  → JSON output    │                    │
│                     └────────┬──────────┘                    │
│                              ▼                               │
│                     ┌───────────────────┐                    │
│                     │  Intent Executor  │                    │
│                     │  (exec1.py)       │                    │
│                     ├───────────────────┤                    │
│                     │ FileOperations    │  create/delete/    │
│                     │   Handler         │  move/copy/rename  │
│                     ├───────────────────┤                    │
│                     │ CodeWriting       │  write & format    │
│                     │   Handler         │  code to file      │
│                     ├───────────────────┤                    │
│                     │ TextProcessing    │  summarize/extract │
│                     │   Handler         │  /analyze          │
│                     └────────┬──────────┘                    │
│                              ▼                               │
│                     ┌───────────────────┐                    │
│                     │  Result Output    │                    │
│                     │  CLI print or     │                    │
│                     │  Streamlit cards  │                    │
│                     └───────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Pipeline, not monolith.** Each stage is independent — you can swap Whisper for another STT, change the LLM, or add new handlers without touching the rest.
- **Strategy pattern for handlers.** `IntentExecutor` dispatches to handler classes via a mapping dict. Adding a new capability means writing one class and registering it.
- **Sandboxed execution.** All file operations are forced into a secure folder. Paths outside the sandbox are rewritten automatically.

---

## File Structure

```
.
├── record.py              # CLI entry point — record, transcribe, classify, execute
├── exec1.py               # Intent executor & handler classes
├── streamlit_app.py       # Streamlit web UI
├── requirements.txt       # Python dependencies (see setup below)
└── README.md
```

| File | Role |
|---|---|
| `record.py` | Captures audio via `sounddevice`, transcribes with Whisper, classifies intent via Ollama, and dispatches to the executor. Standalone CLI tool. |
| `exec1.py` | Defines `IntentExecutor` (dispatcher) and three handler classes: `FileOperationsHandler`, `CodeWritingHandler`, `TextProcessingHandler`. Pure logic — no I/O or UI. |
| `streamlit_app.py` | Web-based console with in-browser recording, audio upload, and text input. Visualizes the four pipeline stages as cards. Imports `exec1.py` for execution. |

---

## Prerequisites

| Dependency | Purpose | Version notes |
|---|---|---|
| **Python** | Runtime | 3.10 – 3.12 recommended (see [Python 3.13 note](#python-313)) |
| **ffmpeg** | Audio decoding/resampling for browser-recorded audio | Any recent version |
| **Ollama** | Local LLM server | Latest stable |
| **PortAudio** | Low-level audio I/O (required by `sounddevice`) | System package |

---

## Setup

### 1. Clone and Install Python Dependencies

```bash
git clone <your-repo-url>
cd voice-ai-agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install \
    sounddevice \
    scipy \
    numpy \
    requests \
    transformers \
    torch \
    streamlit \
    audio-recorder-streamlit \
    pydub
```

Or create a `requirements.txt`:

```
sounddevice
scipy
numpy
requests
transformers
torch
streamlit
audio-recorder-streamlit
pydub
```

```bash
pip install -r requirements.txt
```

### 2. Install System Dependencies

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install ffmpeg portaudio19-dev
```

**macOS (Homebrew):**

```bash
brew install ffmpeg portaudio
```

**Windows:**

- Download ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
- PortAudio is bundled with the `sounddevice` pip package on Windows — no extra install needed.

### 3. Install and Configure Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model used by this project
ollama pull incept5/llama3.1-claude:latest

# Verify it's running
ollama list
```

Ollama serves on `http://localhost:11434` by default. The project hits `/api/chat` — no extra configuration needed.

**Using a different model?** Change `OLLAMA_MODEL` in `record.py` and the sidebar config in `streamlit_app.py`. Any instruction-following model works; smaller models (7B–8B) are fine for intent classification.

### 4. Configure the Secure Folder

All file operations are sandboxed to `/home/bosons/interview/`. Create it or change the path in `exec1.py`:

```bash
mkdir -p /home/bosons/interview/
```

To use a different folder, update the `secure_folder` variable in both `FileOperationsHandler` and `CodeWritingHandler` inside `exec1.py`.

---

## Usage

### Command-Line Mode

```bash
python record.py
```

1. Press **Enter** to start recording.
2. Speak your command.
3. Press **Enter** to stop.
4. The pipeline runs automatically: transcribe → classify → execute.
5. Press **Enter** again to play back the recording.

**Example session:**

```
Press Enter to start recording...
Recording... Press Enter to stop.

Saved to note.wav
Transcribing...

Transcript: Create a Python file called hello.py that prints numbers 1 to 10.

Classifying via ollama...
============================================================
Intent: write code to a new file
Confidence: high
Params: {'filename': 'hello.py', 'body': 'for i in range(1, 11):\n    print(i)'}
============================================================

Executing intent...
============================================================
EXECUTION RESULT:
Status: success
Message: Code written to /home/bosons/interview/hello.py
File saved to: /home/bosons/interview/hello.py
============================================================
```

### Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. The interface has three input tabs:

| Tab | How it works |
|---|---|
| **🔴 Record Audio** | Click the mic icon to record in-browser. Audio is automatically converted from WebM/Opus to 16 kHz WAV before transcription. |
| **📂 Upload Audio** | Drag-and-drop or browse for `.wav`, `.mp3`, `.m4a`, `.ogg` files. Also normalized before transcription. |
| **⌨️ Text Input** | Type a command directly, skipping the audio/STT stages. |

Press **▶ Run** to process. The four pipeline cards update with results. The sidebar holds LLM configuration and a history of past commands.

---

## Hardware Notes & CPU Workarounds

This project was developed and tested entirely on **CPU** — no GPU required. Here's what to expect and how to optimize if you don't have a CUDA-capable GPU.

### Whisper on CPU

The `whisper-base` model (74M parameters) runs comfortably on CPU. Transcription of a 5–10 second command takes roughly 2–4 seconds on a modern laptop.

**Force CPU explicitly** if you have a GPU but want to avoid CUDA issues:

```python
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device="cpu"
)
```

**Install CPU-only PyTorch** to save ~2 GB of disk space (skips CUDA libraries):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Model size vs. speed tradeoffs:**

| Model | Parameters | CPU time (5s audio) | Accuracy |
|---|---|---|---|
| `whisper-tiny` | 39M | ~1–2s | Good for short commands |
| `whisper-base` | 74M | ~2–4s | **Best balance (default)** |
| `whisper-small` | 244M | ~6–10s | Better for accented speech |
| `whisper-large-v3` | 1.5B | ~30–60s | Overkill for commands |

For this use case — short, directed commands in English — `whisper-base` is the sweet spot. Going larger adds latency with negligible accuracy improvement on command-style inputs.

### Ollama on CPU

Ollama runs LLMs on CPU by default when no GPU is detected. For intent classification (short input, short JSON output), even 7B-parameter models respond in 2–5 seconds on CPU.

**Tips for faster CPU inference:**

- Use **quantized models** (Q4_K_M or Q4_0) — Ollama serves these by default.
- Stick to **7B–8B parameter** models. Larger models (13B+) are slower and don't improve classification accuracy for this task.
- Set `"num_ctx": 512` in the Ollama request if you want to limit context window size and reduce memory usage:

```python
requests.post(ENDPOINTS["ollama"], json={
    "model": OLLAMA_MODEL,
    "messages": messages,
    "stream": False,
    "options": {"num_ctx": 512}
})
```

### RAM Requirements

| Component | Estimated RAM |
|---|---|
| Whisper-base | ~300 MB |
| Ollama (7B Q4) | ~4–5 GB |
| Streamlit + Python | ~200 MB |
| **Total** | **~5–6 GB** |

A machine with **8 GB RAM** can run the full stack. 16 GB gives comfortable headroom.

### Apple Silicon (M1/M2/M3/M4)

Ollama automatically uses Metal acceleration on Apple Silicon, which makes LLM inference significantly faster than x86 CPU. Whisper also benefits from the unified memory architecture. If you're on a Mac, this is the best CPU-only experience you'll get.

---

## Supported Intents

| Intent | Handler | Example command |
|---|---|---|
| Create a file | `FileOperationsHandler` | *"Create a file called notes.txt with today's meeting notes"* |
| Delete a file | `FileOperationsHandler` | *"Delete the file old_script.py"* |
| Move / Copy / Rename | `FileOperationsHandler` | *"Rename data.csv to data_backup.csv"* |
| Write code to file | `CodeWritingHandler` | *"Write a Python script called sort.py that implements bubble sort"* |
| Generate code | `CodeWritingHandler` | *"Create a bash script that lists all .log files"* |
| Summarize text | `TextProcessingHandler` | *"Summarize the following paragraph: ..."* |
| Extract entities | `TextProcessingHandler` | *"Extract keywords from this text"* |
| Analyze sentiment | `TextProcessingHandler` | *"Analyze the sentiment of this review"* |
| General chat | — | *"What's the capital of France?"* (classified but no file action) |

### Adding New Intents

1. Create a handler class in `exec1.py` inheriting from `IntentHandler`.
2. Register it in `IntentExecutor.__init__()`:

```python
self.handlers['my_new_intent'] = MyNewHandler()
```

3. Add natural language mappings:

```python
self.intent_mapping['do something new'] = 'my_new_intent'
```

4. Update the LLM system prompt in `record.py` / `streamlit_app.py` to include an example of the new intent.

---

## Configuration

All configuration lives at the top of the source files — no external config files needed.

| Setting | File | Default | Description |
|---|---|---|---|
| `LLM_BACKEND` | `record.py`, `streamlit_app.py` | `"ollama"` | `"ollama"` or `"lmstudio"` |
| `OLLAMA_MODEL` | `record.py`, `streamlit_app.py` | `"incept5/llama3.1-claude:latest"` | Any model pulled in Ollama |
| `SAMPLE_RATE` | `record.py` | `44100` | Microphone sample rate (CLI only) |
| `secure_folder` | `exec1.py` | `"/home/bosons/interview/"` | Sandbox directory for all file operations |

The Streamlit sidebar also exposes the LLM backend and model name for runtime switching.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'audioop'`

**Cause:** Python 3.13 removed the `audioop` module from the standard library. `pydub` depends on it.

**Fix:**

```bash
pip install audioop-lts
```

This installs a drop-in backport. Alternatively, use Python 3.12 or earlier.

### Garbled transcription from browser recording

**Cause:** The browser's MediaRecorder encodes audio as WebM/Opus, not WAV. If raw bytes are written to a `.wav` file, Whisper misinterprets the codec.

**Fix:** This is already handled in `streamlit_app.py` via the `normalize_browser_audio()` function. Make sure `ffmpeg` and `pydub` are installed:

```bash
sudo apt install ffmpeg
pip install pydub
```

### Ollama connection refused

**Cause:** Ollama isn't running.

**Fix:**

```bash
ollama serve          # start the server
ollama list           # verify models are available
```

Check that port 11434 isn't blocked by a firewall.

### `sounddevice` can't find audio device

**Cause:** PortAudio isn't installed.

**Fix:**

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev

# macOS
brew install portaudio

# Then reinstall sounddevice
pip install --force-reinstall sounddevice
```

### Streamlit transcript text is invisible / low contrast

The UI uses a dark theme with solid card backgrounds. If text appears washed out, you may be running a Streamlit version that injects conflicting CSS. The current `streamlit_app.py` uses `!important` overrides and inline styles to force readability. Update to the latest version of the app if you're seeing this.

---

## License

MIT — do whatever you want with it.

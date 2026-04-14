#!/usr/bin/python3
"""
Streamlit Console for Voice-Controlled AI Agent
Records audio → Transcribes → Classifies intent → Executes action
"""

import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import json
import time
import requests
import os
import io
import tempfile
from pathlib import Path
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment

# ─── Page Config ───
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    .app-header h1 {
        color: #e2e8f0;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Pipeline Card ── */
    .pipeline-card {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }
    .pipeline-card:hover {
        border-color: rgba(99,102,241,0.3);
    }
    .pipeline-card.active {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.1);
    }
    .pipeline-card.success {
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    .pipeline-card.error {
        border-color: rgba(239, 68, 68, 0.5);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
    }

    /* ── Step header ── */
    .step-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .step-number {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        font-size: 0.75rem;
        font-weight: 700;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .step-number.done {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
    }
    .step-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* ── Content blocks ── */
    .transcript-text {
        background: rgba(99, 102, 241, 0.12);
        border-left: 3px solid #818cf8;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.25rem;
        color: #f1f5f9;
        font-size: 1rem;
        line-height: 1.7;
        font-style: italic;
    }
    .intent-badge {
        display: inline-block;
        background: rgba(245, 158, 11, 0.18);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 8px;
        padding: 0.35rem 0.85rem;
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .confidence-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .confidence-high {
        background: rgba(16, 185, 129, 0.18);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .confidence-medium {
        background: rgba(245, 158, 11, 0.18);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .confidence-low {
        background: rgba(239, 68, 68, 0.18);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .params-block {
        background: rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #e0e7ff;
        line-height: 1.7;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    /* ── Action ── */
    .action-line {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #e2e8f0;
        font-size: 0.95rem;
        padding: 0.5rem 0;
    }
    .action-line .icon { font-size: 1.1rem; }

    /* ── Result ── */
    .result-success {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #34d399;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.25rem;
        color: #ecfdf5;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .result-error {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #f87171;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.25rem;
        color: #fef2f2;
        font-size: 0.95rem;
    }
    .code-output {
        background: rgba(0,0,0,0.45);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #f8fafc;
        line-height: 1.7;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    /* ── Sidebar ── */
    .config-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }

    /* ── Waiting state ── */
    .waiting {
        color: #94a3b8;
        font-size: 0.9rem;
        font-style: italic;
        padding: 0.5rem 0;
    }

    /* ── History item ── */
    .history-item {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: background 0.2s;
    }
    .history-item:hover {
        background: rgba(99, 102, 241, 0.06);
    }
    .history-time {
        color: #94a3b8;
        font-size: 0.75rem;
    }
    .history-text {
        color: #e2e8f0;
        font-size: 0.85rem;
        margin-top: 0.15rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* ── Recorder ── */
    .recorder-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.5rem 1rem;
        gap: 1rem;
    }
    .recorder-hint {
        color: #cbd5e1;
        font-size: 0.9rem;
        text-align: center;
        line-height: 1.5;
    }
    .recorder-ready {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        color: #34d399;
        font-size: 0.85rem;
        font-weight: 500;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─── State Init ───
defaults = {
    "pipeline_stage": "idle",       # idle | recording | transcribing | classifying | executing | done
    "transcript": None,
    "intent_data": None,
    "action_description": None,
    "result": None,
    "history": [],
    "error": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Backend functions (imported from your code) ───

# LLM Config
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "incept5/llama3.1-claude:latest"
ENDPOINTS = {
    "ollama": "http://localhost:11434/api/chat",
    "lmstudio": "http://localhost:1234/v1/chat/completions",
}

SYSTEM_PROMPT = '''You are an intent classifier and params extractor. 

Examples of intents:
1. Create a file
2. Write code to a new or existing file
3. Summarize a piece of text
4. General Chat

Reply in JSON format:
{
  "intent_id": "1",
  "intent": "create a file",
  "confidence": "high",
  "params": {
    "filename": "example.txt",
    "content": "file content here"
  }
}

For write code intent:
{
  "intent_id": "2",
  "intent": "write code to a new file",
  "confidence": "high",
  "params": {
    "filename": "script.py",
    "body": "actual python code here"
  }
}

Important:
- Extract filename from the user prompt and place in params
- For code writing, include the actual code in the "body" field
- Secure folder: /home/bosons/interview/ (all files will be saved here)
- Confidence can be: high, medium, or low
- Return ONLY valid JSON, no explanations'''


def normalize_browser_audio(raw_bytes: bytes) -> str:
    """
    Convert browser-recorded audio (WebM/Opus/OGG) to a clean
    16-bit 16 kHz mono WAV that Whisper can reliably transcribe.

    Returns the path to the converted temp file (caller must delete).
    """
    # Write raw bytes so pydub/ffmpeg can probe the real codec
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as src:
        src.write(raw_bytes)
        src_path = src.name

    try:
        # Let ffmpeg auto-detect the input format
        audio = AudioSegment.from_file(src_path)
    except Exception:
        # Fallback: try treating as raw WAV in case the component header is fine
        try:
            audio = AudioSegment.from_file(io.BytesIO(raw_bytes), format="wav")
        except Exception:
            os.unlink(src_path)
            raise RuntimeError(
                "Could not decode the recorded audio. "
                "Make sure ffmpeg is installed (sudo apt install ffmpeg)."
            )
    finally:
        if os.path.exists(src_path):
            os.unlink(src_path)

    # Normalize to exactly what Whisper expects
    audio = (
        audio
        .set_channels(1)          # mono
        .set_frame_rate(16000)    # 16 kHz (Whisper's native rate)
        .set_sample_width(2)      # 16-bit
    )

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(out.name, format="wav")
    out.close()
    return out.name


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper."""
    from transformers import pipeline as hf_pipeline
    if "asr_model" not in st.session_state:
        with st.spinner("Loading Whisper model (first run only)..."):
            st.session_state.asr_model = hf_pipeline(
                "automatic-speech-recognition", model="openai/whisper-base"
            )
    result = st.session_state.asr_model(audio_path, language="en")
    return result["text"]


def classify_intent(text: str, backend: str, model: str) -> str:
    """Send text to LLM for intent classification."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    if backend == "ollama":
        resp = requests.post(
            ENDPOINTS["ollama"],
            json={"model": model, "messages": messages, "stream": False, "raw": True, "stop": []},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    elif backend == "lmstudio":
        resp = requests.post(
            ENDPOINTS["lmstudio"],
            json={"model": model, "messages": messages},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unknown backend: {backend}")


def parse_intent(raw_json: str):
    """Parse the intent JSON string into structured data."""
    try:
        # Try to extract JSON from the response (handle markdown fences)
        cleaned = raw_json.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        data = json.loads(cleaned)
        return {
            "intent": data.get("intent", "unknown"),
            "intent_id": data.get("intent_id", "?"),
            "confidence": data.get("confidence", "unknown"),
            "params": data.get("params", {}),
        }
    except (json.JSONDecodeError, IndexError) as e:
        return {"intent": "parse_error", "confidence": "low", "params": {}, "error": str(e), "raw": raw_json}


def execute_intent(intent_str: str, params: dict):
    """Execute the classified intent using IntentExecutor."""
    try:
        from exec1 import IntentExecutor
        executor = IntentExecutor()
        return executor.execute(intent_str, params)
    except Exception as e:
        return {"status": "error", "message": str(e)}


def describe_action(intent_data: dict) -> str:
    """Generate a human-readable description of the action being taken."""
    intent = intent_data.get("intent", "").lower()
    params = intent_data.get("params", {})

    if "file" in intent and "code" not in intent:
        fname = params.get("filename") or params.get("file_path") or "unknown"
        return f"📁  Creating file: {fname}"
    elif "code" in intent or "write" in intent:
        fname = params.get("filename") or params.get("file_path") or "script"
        return f"💻  Writing code to: {fname}"
    elif "summar" in intent:
        return "📝  Summarizing provided text"
    elif "chat" in intent or "general" in intent:
        return "💬  General conversation (no file action)"
    else:
        return f"⚙️  Executing: {intent}"


def run_pipeline(audio_path: str, backend: str, model: str):
    """Run the full pipeline: transcribe → classify → execute."""
    # Stage 1: Transcribe
    st.session_state.pipeline_stage = "transcribing"
    try:
        transcript = transcribe_audio(audio_path)
        st.session_state.transcript = transcript
    except Exception as e:
        st.session_state.error = f"Transcription failed: {e}"
        st.session_state.pipeline_stage = "done"
        return

    # Stage 2: Classify
    st.session_state.pipeline_stage = "classifying"
    try:
        raw = classify_intent(transcript, backend, model)
        intent_data = parse_intent(raw)
        st.session_state.intent_data = intent_data
        st.session_state.action_description = describe_action(intent_data)
    except Exception as e:
        st.session_state.error = f"Classification failed: {e}"
        st.session_state.pipeline_stage = "done"
        return

    # Stage 3: Execute
    st.session_state.pipeline_stage = "executing"
    try:
        if intent_data.get("intent") == "parse_error":
            st.session_state.result = {"status": "error", "message": "Could not parse LLM response", "raw": intent_data.get("raw", "")}
        else:
            result = execute_intent(intent_data["intent"], intent_data["params"])
            st.session_state.result = result
    except Exception as e:
        st.session_state.error = f"Execution failed: {e}"

    st.session_state.pipeline_stage = "done"

    # Save to history
    st.session_state.history.insert(0, {
        "time": time.strftime("%H:%M:%S"),
        "transcript": st.session_state.transcript,
        "intent": intent_data.get("intent", "?"),
        "status": st.session_state.result.get("status", "error") if st.session_state.result else "error",
    })


def run_text_pipeline(text: str, backend: str, model: str):
    """Run the pipeline from text input (skip transcription)."""
    st.session_state.transcript = text
    st.session_state.pipeline_stage = "classifying"

    try:
        raw = classify_intent(text, backend, model)
        intent_data = parse_intent(raw)
        st.session_state.intent_data = intent_data
        st.session_state.action_description = describe_action(intent_data)
    except Exception as e:
        st.session_state.error = f"Classification failed: {e}"
        st.session_state.pipeline_stage = "done"
        return

    st.session_state.pipeline_stage = "executing"
    try:
        if intent_data.get("intent") == "parse_error":
            st.session_state.result = {"status": "error", "message": "Could not parse LLM response", "raw": intent_data.get("raw", "")}
        else:
            result = execute_intent(intent_data["intent"], intent_data["params"])
            st.session_state.result = result
    except Exception as e:
        st.session_state.error = f"Execution failed: {e}"

    st.session_state.pipeline_stage = "done"

    st.session_state.history.insert(0, {
        "time": time.strftime("%H:%M:%S"),
        "transcript": text,
        "intent": intent_data.get("intent", "?") if 'intent_data' in dir() else "?",
        "status": st.session_state.result.get("status", "error") if st.session_state.result else "error",
    })


def reset_pipeline():
    st.session_state.pipeline_stage = "idle"
    st.session_state.transcript = None
    st.session_state.intent_data = None
    st.session_state.action_description = None
    st.session_state.result = None
    st.session_state.error = None


# ─── Sidebar ───
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("")

    backend = st.selectbox("LLM Backend", ["ollama", "lmstudio"], index=0)
    model_name = st.text_input("Model Name", value=OLLAMA_MODEL)
    st.markdown("---")

    st.markdown("### 📜 History")
    if not st.session_state.history:
        st.caption("No commands yet.")
    for item in st.session_state.history[:10]:
        status_icon = "✅" if item["status"] == "success" else "❌"
        st.markdown(
            f"""<div class="history-item">
                <div class="history-time">{item['time']}  {status_icon}</div>
                <div class="history-text">{item['transcript'][:60]}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ─── Header ───
st.markdown("""
<div class="app-header">
    <h1>🎙️ Voice AI Agent Console</h1>
    <p>Record or type a command → Transcribe → Classify Intent → Execute Action</p>
</div>
""", unsafe_allow_html=True)


# ─── Input Section ───
col_input, col_spacer, col_action = st.columns([5, 0.5, 2])

recorded_bytes = None
uploaded = None
text_input = ""

with col_input:
    tab_record, tab_audio, tab_text = st.tabs(["🔴  Record Audio", "📂  Upload Audio", "⌨️  Text Input"])

    with tab_record:
        st.markdown('<div class="recorder-container">', unsafe_allow_html=True)
        st.markdown('<div class="recorder-hint">Click the microphone to start recording.<br>Click again to stop. Audio appears below when ready.</div>', unsafe_allow_html=True)
        recorded_bytes = audio_recorder(
            text="",
            recording_color="#ef4444",
            neutral_color="#6366f1",
            icon_size="2x",
            pause_threshold=300.0,   # very long so it won't auto-stop
            sample_rate=44100,
            key="voice_recorder",
        )
        if recorded_bytes:
            st.markdown('<div class="recorder-ready">✓ Recording captured — press <b>Run</b> to process</div>', unsafe_allow_html=True)
            st.audio(recorded_bytes, format="audio/wav")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_audio:
        uploaded = st.file_uploader(
            "Upload a voice recording (.wav, .mp3, .m4a, .ogg)",
            type=["wav", "mp3", "m4a", "ogg"],
            label_visibility="collapsed",
        )
        st.caption("Record using your device's recorder app, then upload the file here.")

    with tab_text:
        text_input = st.text_area(
            "Type your command directly",
            placeholder="e.g. Create a Python file called hello.py that prints numbers 1 to 10",
            height=100,
            label_visibility="collapsed",
        )

with col_action:
    st.markdown("<div style='height: 42px'></div>", unsafe_allow_html=True)
    col_run, col_reset = st.columns(2)
    with col_run:
        run_clicked = st.button("▶  Run", use_container_width=True, type="primary")
    with col_reset:
        reset_clicked = st.button("↺  Reset", use_container_width=True)

if reset_clicked:
    reset_pipeline()
    st.rerun()

if run_clicked:
    reset_pipeline()
    if recorded_bytes:
        # Decode browser audio (WebM/Opus) → clean 16 kHz mono WAV
        try:
            tmp_path = normalize_browser_audio(recorded_bytes)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        run_pipeline(tmp_path, backend, model_name)
        os.unlink(tmp_path)
    elif uploaded is not None:
        # Save uploaded file, then normalize to clean WAV
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            raw_path = tmp.name
        try:
            audio = AudioSegment.from_file(raw_path)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            tmp_path = raw_path.rsplit(".", 1)[0] + "_norm.wav"
            audio.export(tmp_path, format="wav")
            os.unlink(raw_path)
        except Exception:
            # Fallback: use the file as-is if conversion fails
            tmp_path = raw_path
        run_pipeline(tmp_path, backend, model_name)
        os.unlink(tmp_path)
    elif text_input.strip():
        run_text_pipeline(text_input.strip(), backend, model_name)
    else:
        st.warning("Please record audio, upload a file, or type a command first.")
        st.stop()
    st.rerun()


# ─── Pipeline Display ───
st.markdown("")

stage = st.session_state.pipeline_stage

# ---------- Step 1: Transcription ----------
card_class = "pipeline-card"
if st.session_state.transcript:
    card_class += " success"

num_class = "step-number done" if st.session_state.transcript else "step-number"
st.markdown(f"""
<div class="{card_class}">
    <div class="step-header">
        <div class="{num_class}">1</div>
        <div class="step-title">Transcription</div>
    </div>
""", unsafe_allow_html=True)

if st.session_state.transcript:
    st.markdown(f'<div class="transcript-text">"{st.session_state.transcript}"</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="waiting">Waiting for input…</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ---------- Step 2: Intent Classification ----------
card_class = "pipeline-card"
if st.session_state.intent_data:
    card_class += " success" if st.session_state.intent_data.get("intent") != "parse_error" else " error"

num_class = "step-number done" if st.session_state.intent_data else "step-number"
st.markdown(f"""
<div class="{card_class}">
    <div class="step-header">
        <div class="{num_class}">2</div>
        <div class="step-title">Intent Classification</div>
    </div>
""", unsafe_allow_html=True)

if st.session_state.intent_data:
    idata = st.session_state.intent_data
    conf = idata.get("confidence", "unknown")
    conf_class = f"confidence-{conf}" if conf in ("high", "medium", "low") else "confidence-low"

    st.markdown(f"""
        <div>
            <span class="intent-badge">{idata.get('intent', 'unknown')}</span>
            <span class="confidence-badge {conf_class}">{conf} confidence</span>
        </div>
    """, unsafe_allow_html=True)

    if idata.get("params"):
        st.markdown(f'<div class="params-block">{json.dumps(idata["params"], indent=2)}</div>', unsafe_allow_html=True)

    if idata.get("raw"):
        with st.expander("Raw LLM Response"):
            st.code(idata["raw"], language="json")
else:
    st.markdown('<div class="waiting">Waiting for transcription…</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ---------- Step 3: Action Taken ----------
card_class = "pipeline-card"
if st.session_state.action_description:
    card_class += " active"

num_class = "step-number done" if st.session_state.action_description else "step-number"
st.markdown(f"""
<div class="{card_class}">
    <div class="step-header">
        <div class="{num_class}">3</div>
        <div class="step-title">Action Taken</div>
    </div>
""", unsafe_allow_html=True)

if st.session_state.action_description:
    st.markdown(f'<div class="action-line">{st.session_state.action_description}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="waiting">Waiting for classification…</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ---------- Step 4: Result ----------
card_class = "pipeline-card"
result = st.session_state.result
if result:
    card_class += " success" if result.get("status") == "success" else " error"

num_class = "step-number done" if result else "step-number"
st.markdown(f"""
<div class="{card_class}">
    <div class="step-header">
        <div class="{num_class}">4</div>
        <div class="step-title">Result</div>
    </div>
""", unsafe_allow_html=True)

if result:
    if result.get("status") == "success":
        st.markdown(f'<div class="result-success">✅  {result.get("message", "Done")}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-error">❌  {result.get("message", "Unknown error")}</div>', unsafe_allow_html=True)

    if result.get("code"):
        st.markdown('<div style="margin-top:0.75rem; color:#94a3b8; font-size:0.82rem; font-weight:600;">GENERATED CODE</div>', unsafe_allow_html=True)
        st.code(result["code"], language="python")

    if result.get("file_path"):
        st.markdown(f"""
            <div class="action-line" style="margin-top:0.5rem;">
                <span class="icon">📄</span> Saved to: <code>{result["file_path"]}</code>
            </div>
        """, unsafe_allow_html=True)

    if result.get("result"):
        st.markdown(f'<div class="params-block">{result["result"]}</div>', unsafe_allow_html=True)

elif st.session_state.error:
    st.markdown(f'<div class="result-error">❌  {st.session_state.error}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="waiting">Waiting for execution…</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

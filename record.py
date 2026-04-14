#!/usr/bin/python3
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import threading
import requests
import json
from transformers import pipeline
from exec1 import IntentExecutor

SAMPLE_RATE = 44100
FILENAME = "note.wav"

# --- Config: switch between Ollama and LM Studio ---
LLM_BACKEND = "ollama"        # "ollama" or "lmstudio"
OLLAMA_MODEL = "incept5/llama3.1-claude:latest"     # any model you have pulled

ENDPOINTS = {
    "ollama":   "http://localhost:11434/api/chat",
    "lmstudio": "http://localhost:1234/v1/chat/completions",
}

system_prompt= 'You are an intent classifier and params extractor. Examples of intents are 1. Create a file ; 2. Write code to a new or existing file ; 3. Summarize a piece of text; 4. General Chat; There could be other intents; Reply in Json format: {"intent_id":"1", "intent":"create a file", "confidence": "high", "params": "[placeholder for filename mentioned in the prompt]"}; Similarly for other file operations filename to be extracted from the user prompt and places in "params" field of the Output JSON. Secure folder: /home/bosons/interview/ to be utilized for reading / writing files . No other folder is considered safe.  confidence can be high, low or medium depending on the strength of the conclusion reached by the assistant or agent. Please do not include any explanation. only the JSON output'


def classify_intent(text):
    messages = [{"role":"system", "content": system_prompt},{ "role": "user", "content": text }]

    if LLM_BACKEND == "ollama":
        response = requests.post(
            ENDPOINTS["ollama"],
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,"raw":True, "stop": []},
        )
        print(response.text)
        raw = response.json()["message"]["content"]
    else:
        raise ValueError(f"Unknown backend: {LLM_BACKEND}")

    return raw

print("Loading Whisper model...")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")

chunks = []
stop_event = threading.Event()

def callback(indata, frames, time, status):
    if not stop_event.is_set():
        chunks.append(indata.copy())

if __name__ == "__main__":
    input("Press Enter to start recording...")
    print("Recording... Press Enter to stop.")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input()                # blocks until Enter
        stop_event.set()       # signals callback to stop collecting

    audio = np.concatenate(chunks)
    wav.write(FILENAME, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"Saved to {FILENAME}")

    print("Transcribing...")
    transcript = asr(FILENAME, language='en')
    print(f"\nTranscript: {transcript['text']}")

    print(f"\nClassifying via {LLM_BACKEND}...")
    intent = classify_intent(transcript['text'])
    print(intent)
    intent2 = json.loads(intent)
    print(intent2)
    intent_type = intent2["intent"]
    params = json.loads(intent)["params"]

    input("Press Enter to play back...")
    rate, data = wav.read(FILENAME)
    sd.play(data, rate)
    sd.wait()
    print("Done.")

    executor = IntentExecutor()
    result = executor.execute(intent_type, params)

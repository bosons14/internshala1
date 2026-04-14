#!/usr/bin/python3
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import threading
import requests
import json
from transformers import pipeline
from exec1 import IntentExecutor  # Use the improved version

SAMPLE_RATE = 44100
FILENAME = "note.wav"

# --- Config: switch between Ollama and LM Studio ---
LLM_BACKEND = "ollama"        # "ollama" or "lmstudio"
OLLAMA_MODEL = "incept5/llama3.1-claude:latest"     # any model you have pulled

ENDPOINTS = {
    "ollama":   "http://localhost:11434/api/chat",
    "lmstudio": "http://localhost:1234/v1/chat/completions",
}

system_prompt = '''You are an intent classifier and params extractor. 

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


def classify_intent(text):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    if LLM_BACKEND == "ollama":
        response = requests.post(
            ENDPOINTS["ollama"],
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "raw": True,
                "stop": []
            },
        )
        print("Raw Ollama response:")
        print(response.text)
        print()
        
        raw = response.json()["message"]["content"]
        return raw
    else:
        raise ValueError(f"Unknown backend: {LLM_BACKEND}")


def parse_intent_response(intent_json_str):
    """
    Parse the intent JSON response, handling both formats:
    1. Proper JSON with nested params dict
    2. String params like your LLM is currently returning
    """
    try:
        intent_data = json.loads(intent_json_str)
        
        intent = intent_data.get("intent", "")
        params = intent_data.get("params", {})
        confidence = intent_data.get("confidence", "unknown")
        
        print(f"\n{'='*60}")
        print(f"Intent: {intent}")
        print(f"Confidence: {confidence}")
        print(f"Params: {params}")
        print(f"{'='*60}\n")
        
        return intent, params
        
    except json.JSONDecodeError as e:
        print(f"Error parsing intent JSON: {e}")
        print(f"Raw response: {intent_json_str}")
        return None, None


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
    print(f"\nTranscript: {transcript['text']}\n")

    print(f"Classifying via {LLM_BACKEND}...")
    intent_json = classify_intent(transcript['text'])
    
    # Parse the intent response
    intent, params = parse_intent_response(intent_json)
    
    if intent and params:
        # Execute the intent
        print("Executing intent...")
        executor = IntentExecutor()
        result = executor.execute(intent, params)
        
        print(f"\n{'='*60}")
        print("EXECUTION RESULT:")
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        if 'code' in result:
            print(f"\nGenerated Code:\n{result.get('code')}")
        if 'file_path' in result:
            print(f"\nFile saved to: {result.get('file_path')}")
        print(f"{'='*60}\n")
    else:
        print("Failed to parse intent. Skipping execution.")

    input("Press Enter to play back recording...")
    rate, data = wav.read(FILENAME)
    sd.play(data, rate)
    sd.wait()
    print("Done.")

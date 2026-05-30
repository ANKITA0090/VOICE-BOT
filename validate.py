"""
Run this to validate your full setup before running realtime.py
    python validate.py
"""

import asyncio
import base64
import json
import os
import sys
import wave
import struct
import tempfile

from dotenv import load_dotenv
load_dotenv()

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

api_key = os.environ.get("OPENAI_API_KEY", "")


# ── 1. API Key ────────────────────────────────────────────────────────────────
def check_api_key():
    print("\n--- 1. API Key ---")
    if not api_key:
        print(f"{FAIL} OPENAI_API_KEY not set in .env")
        return False
    if not api_key.startswith("sk-"):
        print(f"{FAIL} Key doesn't look right (should start with sk-)")
        return False
    print(f"{PASS} Key found: {api_key[:12]}...")
    return True


# ── 2. OpenAI connectivity + list available models ────────────────────────────
def check_openai_models():
    print("\n--- 2. Available OpenAI Models ---")
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        models = [m.id for m in client.models.list().data]
        models.sort()

        # Check ones we care about
        targets = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-realtime-preview",
            "gpt-4o-realtime-preview-2024-12-17",
            "gpt-4o-mini-realtime-preview",
            "gpt-4o-mini-realtime-preview-2024-12-17",
            "whisper-1",
            "tts-1",
            "text-embedding-3-small",
        ]

        found_realtime = []
        for t in targets:
            if t in models:
                print(f"  {PASS} {t}")
                if "realtime" in t:
                    found_realtime.append(t)
            else:
                print(f"  {FAIL} {t}  ← not on your account")

        return found_realtime
    except Exception as e:
        print(f"{FAIL} Could not reach OpenAI: {e}")
        return []


# ── 3. Realtime WebSocket ─────────────────────────────────────────────────────
async def _try_realtime(model):
    from websockets.asyncio.client import connect
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    try:
        async with connect(url, additional_headers={"Authorization": f"Bearer {api_key}"}) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"modalities": ["text"], "instructions": "test"}
            }))
            async for raw in ws:
                evt = json.loads(raw)
                if evt.get("type") == "session.updated":
                    return True
                if evt.get("type") == "error":
                    return False
    except Exception:
        return False
    return False


def check_realtime(realtime_models):
    print("\n--- 3. Realtime WebSocket ---")
    if not realtime_models:
        print(f"{SKIP} No realtime models on your account — skipping")
        return None

    working = None
    for model in realtime_models:
        ok = asyncio.run(_try_realtime(model))
        if ok:
            print(f"  {PASS} {model}  ← use this in config.py")
            working = model
        else:
            print(f"  {FAIL} {model}")

    return working


# ── 4. Microphone ─────────────────────────────────────────────────────────────
def check_microphone():
    print("\n--- 4. Microphone ---")
    try:
        import sounddevice as sd
        import numpy as np

        devices = sd.query_devices()
        inputs  = [d for d in devices if d["max_input_channels"] > 0]
        if not inputs:
            print(f"{FAIL} No microphone found")
            return False

        default = sd.query_devices(kind="input")
        print(f"  {PASS} Default input: {default['name']}")

        # Quick 0.5s recording test
        rec = sd.rec(int(0.5 * 24000), samplerate=24000, channels=1, dtype="int16")
        sd.wait()
        if rec is not None and len(rec) > 0:
            print(f"  {PASS} Recording works (captured {len(rec)} frames)")
            return True
        else:
            print(f"{FAIL} Recording returned empty data")
            return False
    except Exception as e:
        print(f"{FAIL} Microphone error: {e}")
        return False


# ── 5. Speaker / Playback ─────────────────────────────────────────────────────
def check_speaker():
    print("\n--- 5. Speaker ---")
    try:
        import sounddevice as sd
        import numpy as np

        default = sd.query_devices(kind="output")
        print(f"  {PASS} Default output: {default['name']}")

        # Play a short 440Hz beep
        t    = np.linspace(0, 0.3, int(0.3 * 24000), False)
        beep = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        sd.play(beep, 24000)
        sd.wait()
        print(f"  {PASS} Playback works (you should have heard a beep)")
        return True
    except Exception as e:
        print(f"{FAIL} Speaker error: {e}")
        return False


# ── 6. Whisper STT ────────────────────────────────────────────────────────────
def check_whisper():
    print("\n--- 6. Whisper STT ---")
    try:
        import openai

        # Create a minimal silent WAV in memory
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 8000, *([0] * 8000)))

        client = openai.OpenAI(api_key=api_key)
        with open(tmp.name, "rb") as f:
            result = client.audio.transcriptions.create(model="whisper-1", file=f)
        print(f"  {PASS} Whisper API reachable (transcribed silent audio)")
        return True
    except Exception as e:
        print(f"{FAIL} Whisper error: {e}")
        return False


# ── 7. ChromaDB / Knowledge Base ──────────────────────────────────────────────
def check_chromadb():
    print("\n--- 7. ChromaDB Knowledge Base ---")
    from config import DB_FOLDER, EMBED_MODEL, TOP_K_RESULTS
    if not os.path.exists(DB_FOLDER):
        print(f"  {SKIP} chroma_db/ not found — run ingest.py first")
        return False
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma

        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        db         = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
        count      = db._collection.count()
        print(f"  {PASS} ChromaDB loaded — {count} chunks stored")

        results = db.similarity_search("test", k=1)
        print(f"  {PASS} Search works — retrieved {len(results)} result(s)")
        return True
    except Exception as e:
        print(f"{FAIL} ChromaDB error: {e}")
        return False


# ── 8. Config sanity check ────────────────────────────────────────────────────
def check_config():
    print("\n--- 8. Config ---")
    try:
        from config import (
            SYSTEM_PROMPT, REALTIME_MODEL, LLM_MODEL, TTS_VOICE,
            TEMPERATURE, VAD_THRESHOLD, VAD_SILENCE_MS
        )
        print(f"  {PASS} REALTIME_MODEL   = {REALTIME_MODEL}")
        print(f"  {PASS} LLM_MODEL        = {LLM_MODEL}")
        print(f"  {PASS} TTS_VOICE        = {TTS_VOICE}")
        print(f"  {PASS} TEMPERATURE      = {TEMPERATURE}")
        print(f"  {PASS} VAD_THRESHOLD    = {VAD_THRESHOLD}")
        print(f"  {PASS} VAD_SILENCE_MS   = {VAD_SILENCE_MS}")
        print(f"  {PASS} SYSTEM_PROMPT    = {len(SYSTEM_PROMPT)} chars")
        return True
    except Exception as e:
        print(f"{FAIL} Config error: {e}")
        return False


# ── Summary ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Voice Bot — Full Setup Validation")
    print("=" * 50)

    results = {}
    results["api_key"]    = check_api_key()

    if not results["api_key"]:
        print("\nCannot continue without a valid API key.")
        sys.exit(1)

    realtime_models       = check_openai_models()
    working_rt            = check_realtime(realtime_models)
    results["mic"]        = check_microphone()
    results["speaker"]    = check_speaker()
    results["whisper"]    = check_whisper()
    results["chromadb"]   = check_chromadb()
    results["config"]     = check_config()

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    for k, v in results.items():
        icon = PASS if v else (SKIP if v is None else FAIL)
        print(f"  {icon} {k}")

    if working_rt:
        print(f"\nSet this in config.py:")
        print(f'  REALTIME_MODEL = "{working_rt}"')
    elif realtime_models:
        print(f"\nNo realtime model connected — check your account access.")
    else:
        print(f"\nNo realtime models on your account.")
        print(f"  → Use run.py (STT + GPT + TTS pipeline) instead.")

    print()


if __name__ == "__main__":
    main()

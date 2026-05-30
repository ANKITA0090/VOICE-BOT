import asyncio
import base64
import json
import os
import threading
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from websockets.asyncio.client import connect

from config import (
    SYSTEM_PROMPT, REALTIME_MODEL, TTS_VOICE, TTS_SPEED,
    VAD_THRESHOLD, VAD_PREFIX_MS, VAD_SILENCE_MS,
    DATA_FOLDER,
)

load_dotenv()

RATE     = 24000
CHANNELS = 1
CHUNK    = 2400  # 100ms

WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

_buf      = np.array([], dtype=np.int16)
_buf_lock = threading.Lock()


# ── Load full knowledge base into prompt at startup ───────────────────────────
def _load_kb_into_prompt() -> str:
    if not os.path.exists(DATA_FOLDER):
        return SYSTEM_PROMPT

    texts = []
    for fname in os.listdir(DATA_FOLDER):
        if fname.endswith(".txt"):
            path = os.path.join(DATA_FOLDER, fname)
            with open(path, encoding="utf-8") as f:
                texts.append(f.read().strip())

    if not texts:
        return SYSTEM_PROMPT

    kb = "\n\n---\n\n".join(texts)
    print(f"Knowledge base loaded: {len(kb)} characters from {len(texts)} file(s).")
    return f"{SYSTEM_PROMPT}\n\n=== KNOWLEDGE BASE ===\n{kb}\n=== END OF KNOWLEDGE BASE ==="


# ── Playback callback ─────────────────────────────────────────────────────────
def _playback_callback(outdata, frames, time_info, status):
    global _buf
    with _buf_lock:
        if len(_buf) >= frames:
            outdata[:, 0] = _buf[:frames]
            _buf = _buf[frames:]
        else:
            outdata[:len(_buf), 0] = _buf
            outdata[len(_buf):, 0] = 0
            _buf = np.array([], dtype=np.int16)


# ── Main ──────────────────────────────────────────────────────────────────────
async def run():
    global _buf

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    loop = asyncio.get_running_loop()

    # Knowledge base is baked into the system prompt — zero query-time delay
    full_prompt = _load_kb_into_prompt()

    print(f"Connecting to OpenAI Realtime ({REALTIME_MODEL})...")

    async with connect(WS_URL, additional_headers={"Authorization": f"Bearer {api_key}"}) as ws:

        session_cfg = {
            "type": "realtime",
            "instructions": full_prompt,
            "audio": {
                "input": {
                    "transcription": {"model": "gpt-realtime-whisper"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": VAD_THRESHOLD,
                        "prefix_padding_ms": VAD_PREFIX_MS,
                        "silence_duration_ms": VAD_SILENCE_MS,
                        "create_response": True,   # instant — no mid-query delay
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "voice": TTS_VOICE,
                    "speed": TTS_SPEED,
                },
            },
        }

        await ws.send(json.dumps({"type": "session.update", "session": session_cfg}))

        async for raw in ws:
            evt = json.loads(raw)
            if evt.get("type") == "session.updated":
                print("Ready! Just speak. Ctrl+C to quit.\n")
                break
            if evt.get("type") == "error":
                raise RuntimeError(f"Session error: {evt.get('error', {}).get('message')}")

        mic_q: asyncio.Queue = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(mic_q.put_nowait, bytes(indata))

        async def send_mic():
            with sd.InputStream(samplerate=RATE, channels=CHANNELS,
                                dtype="int16", blocksize=CHUNK,
                                callback=mic_callback):
                while True:
                    data = await mic_q.get()
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode(),
                    }))

        async def receive_and_play():
            global _buf
            with sd.OutputStream(samplerate=RATE, channels=CHANNELS,
                                  dtype="int16", blocksize=CHUNK,
                                  callback=_playback_callback):
                async for raw in ws:
                    evt = json.loads(raw)
                    t   = evt.get("type", "")

                    if t == "input_audio_buffer.speech_started":
                        with _buf_lock:
                            _buf = np.array([], dtype=np.int16)
                        print("\nYou: ", end="", flush=True)

                    elif t == "conversation.item.input_audio_transcription.completed":
                        print(evt.get("transcript", "").strip())
                        print("Bot: ", end="", flush=True)

                    elif t == "response.output_audio.delta":
                        pcm = np.frombuffer(
                            base64.b64decode(evt["delta"]), dtype=np.int16
                        )
                        with _buf_lock:
                            _buf = np.concatenate([_buf, pcm])

                    elif t == "response.output_audio_transcript.delta":
                        print(evt.get("delta", ""), end="", flush=True)

                    elif t == "response.done":
                        print()

                    elif t == "error":
                        msg = evt.get("error", {}).get("message", "unknown")
                        print(f"\nAPI Error: {msg}")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(send_mic())
            tg.create_task(receive_and_play())


async def main():
    while True:
        try:
            await run()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nConnection lost: {e}")
            print("Reconnecting in 3 seconds...\n")
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())

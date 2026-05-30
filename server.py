import asyncio
import base64
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect as ws_connect

load_dotenv()

from config import (
    SYSTEM_PROMPT, REALTIME_MODEL, TTS_VOICE, TTS_SPEED,
    VAD_THRESHOLD, VAD_PREFIX_MS, VAD_SILENCE_MS,
    DATA_FOLDER,
)

# ── Load knowledge base once at startup ──────────────────────────────────────
def _build_prompt() -> str:
    if not os.path.exists(DATA_FOLDER):
        return SYSTEM_PROMPT
    texts = []
    for fname in sorted(os.listdir(DATA_FOLDER)):
        if fname.endswith(".txt"):
            with open(os.path.join(DATA_FOLDER, fname), encoding="utf-8") as f:
                texts.append(f.read().strip())
    if not texts:
        return SYSTEM_PROMPT
    kb = "\n\n---\n\n".join(texts)
    print(f"Knowledge base loaded: {sum(len(t) for t in texts):,} chars from {len(texts)} file(s).")
    return f"{SYSTEM_PROMPT}\n\n=== KNOWLEDGE BASE ===\n{kb}\n=== END OF KNOWLEDGE BASE ==="


FULL_PROMPT = _build_prompt()
WS_URL      = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

app = FastAPI()


# ── WebSocket endpoint — one per browser tab ──────────────────────────────────
@app.websocket("/ws")
async def handle(browser: WebSocket):
    await browser.accept()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        await browser.send_json({"type": "error", "message": "OPENAI_API_KEY not set on server."})
        return

    try:
        async with ws_connect(WS_URL, additional_headers={"Authorization": f"Bearer {api_key}"}) as openai:

            # Configure session
            await openai.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": FULL_PROMPT,
                    "audio": {
                        "input": {
                            "transcription": {"model": "gpt-realtime-whisper"},
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": VAD_THRESHOLD,
                                "prefix_padding_ms": VAD_PREFIX_MS,
                                "silence_duration_ms": VAD_SILENCE_MS,
                                "create_response": True,
                                "interrupt_response": True,
                            },
                        },
                        "output": {
                            "voice": TTS_VOICE,
                            "speed": TTS_SPEED,
                        },
                    },
                },
            }))

            # Wait for session confirmation
            async for raw in openai:
                evt = json.loads(raw)
                if evt.get("type") == "session.updated":
                    break
                if evt.get("type") == "error":
                    await browser.send_json({"type": "error", "message": evt["error"]["message"]})
                    return

            await browser.send_json({"type": "ready"})

            # ── Task 1: browser mic → OpenAI ──────────────────────────────
            async def mic_to_openai():
                while True:
                    data = await browser.receive_bytes()
                    await openai.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode(),
                    }))

            # ── Task 2: OpenAI events → browser ──────────────────────────
            async def openai_to_browser():
                async for raw in openai:
                    evt = json.loads(raw)
                    t   = evt.get("type", "")

                    if t == "input_audio_buffer.speech_started":
                        await browser.send_json({"type": "speech_started"})

                    elif t == "conversation.item.input_audio_transcription.completed":
                        await browser.send_json({
                            "type": "user_transcript",
                            "text": evt.get("transcript", "").strip(),
                        })

                    elif t == "response.output_audio.delta":
                        await browser.send_json({
                            "type": "audio",
                            "delta": evt["delta"],
                        })

                    elif t == "response.output_audio_transcript.delta":
                        await browser.send_json({
                            "type": "bot_transcript",
                            "delta": evt.get("delta", ""),
                        })

                    elif t == "response.done":
                        await browser.send_json({"type": "response_done"})

                    elif t == "error":
                        await browser.send_json({
                            "type": "error",
                            "message": evt.get("error", {}).get("message", "unknown"),
                        })

            await asyncio.gather(
                mic_to_openai(),
                openai_to_browser(),
                return_exceptions=True,
            )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await browser.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── Serve frontend ────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")

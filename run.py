from dotenv import load_dotenv
load_dotenv()

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile

from stt import transcribe
from tts import speak
from chain import VoiceBot

SAMPLE_RATE = 16000

bot = VoiceBot()


def record() -> str:
    print("\nListening... press ENTER to stop.")
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()

    audio = np.concatenate(frames, axis=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name


def main():
    print("=" * 40)
    print("  Voice Bot — RAG + GPT")
    print("=" * 40)
    print("Press ENTER to start talking. Type 'quit' to exit.\n")

    while True:
        cmd = input("Press ENTER to speak ('quit' to exit): ").strip().lower()
        if cmd == "quit":
            print("Goodbye!")
            break

        audio_path = record()

        print("Transcribing...")
        user_text = transcribe(audio_path)
        if not user_text:
            print("Could not understand. Try again.\n")
            continue

        print(f"You: {user_text}")
        print("Thinking...")

        reply = bot.chat(user_text)
        print(f"Bot: {reply}\n")

        speak(reply)


if __name__ == "__main__":
    main()

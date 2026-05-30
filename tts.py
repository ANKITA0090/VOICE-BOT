import openai
import tempfile
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

from config import TTS_VOICE


def speak(text: str):
    client = openai.OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    response.stream_to_file(tmp.name)

    rate, data = wav.read(tmp.name)
    sd.play(data, rate)
    sd.wait()

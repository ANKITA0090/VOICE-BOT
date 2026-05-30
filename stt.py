import openai


def transcribe(audio_path: str) -> str:
    client = openai.OpenAI()
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
        )
    return result.text.strip()

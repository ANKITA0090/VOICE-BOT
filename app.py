from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from stt import transcribe
from tts import speak
from chain import VoiceBot

bot = VoiceBot()


def process(audio):
    if audio is None:
        return None
    user_text = transcribe(audio)
    if not user_text:
        return None
    bot_text = bot.chat(user_text)
    return speak(bot_text)


with gr.Blocks(title="Voice Bot") as demo:
    gr.Markdown("## Voice Bot\nPress **Record** to start, press again to stop — response plays automatically.")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="")
    out = gr.Audio(autoplay=True, visible=False)
    mic.stop_recording(fn=process, inputs=[mic], outputs=[out])

demo.launch()

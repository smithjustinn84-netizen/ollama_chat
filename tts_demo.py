import gradio as gr
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

def tts_fn(text, speaker):
    wav_path = "temp/output.wav"
    tts.tts_to_file(text=text, speaker=speaker, language="en", file_path=wav_path)
    return wav_path


demo = gr.Interface(
    fn=tts_fn,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Dropdown(choices=tts.speakers, label="Speaker"),
    ],
    outputs=gr.Audio(label="Generated Audio"),
    title="Text-to-Speech Demo",
    description="Enter text and select a speaker to generate speech.",
)
demo.launch()
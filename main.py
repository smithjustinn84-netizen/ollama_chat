from fastapi import FastAPI
import gradio as gr
import ollama

from gradio import ChatMessage

app = FastAPI()

# Modify the model to what you want
model = "llama3"

@app.get("/")
def read_main():
    return {"message": "This is your main app"}


def sepia(input_img):
    return input_img

client = ollama.Client()

def stream_chat(message: str, history: list, temperature: float, max_new_tokens: int, top_p: float, top_k: int,
                penalty: float):
    conversation = []
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])
    conversation.append({"role": "user", "content": message})

    print(f"Conversation is -\n{conversation}")

    response = client.chat(
        model=model,
        messages=conversation,
        stream=True,
        options={
            'num_predict': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repeat_penalty': penalty,
            'low_vram': True,
        },
    )

    buffer = ""
    for chunk in response:
        buffer += chunk["message"]["content"]
        yield buffer


with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as io:
    with gr.Sidebar(position="left"):
        gr.Markdown("# Advanced Options")
        gr.Markdown("Use the options below to create advanced prompts")
        top_k = gr.Slider(0.0,100.0, label="top_k", value=40, info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
        top_p = gr.Slider(0.0,1.0, label="top_p", value=0.9, info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
        temp = gr.Slider(0.0,2.0, label="temperature", value=0.8, info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)")

    gr.Interface(
        sepia,
        gr.Image(label="input"),
        gr.Image(label="output"),
        title="LLM Image Chat",
        description="Let's talk about the provided image.",
        article="Ask about the image below?",
        css="footer {visibility: hidden}",
        flagging_mode="never"
    )

    gr.ChatInterface(
        stream_chat,
        type="messages",
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.8,
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=128,
                maximum=2048,
                step=1,
                value=1024,
                label="Max New Tokens",
                render=False,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.8,
                label="top_p",
                render=False,
            ),
            gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=20,
                label="top_k",
                render=False,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="Repetition penalty",
                render=False,
            ),
        ],
    )



app = gr.mount_gradio_app(app, io, path="/image")

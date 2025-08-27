from fastapi import FastAPI
import gradio as gr
import ollama
from gradio import FileData
from gradio.components.chatbot import ChatMessage

from utils.utils import image_to_base64

app = FastAPI()

client = ollama.Client()
model_list = ollama.list()
model_names = [model['model'] for model in model_list['models']]


def stream_chat(message: dict | ChatMessage | None, history: list, img: str | None, temperature: float, max_new_tokens: int,
                top_p: float, top_k: int,
                penalty: float, model: str = 'llava'):
    conversation = format_history(history, message)

    if img is not None:
        conversation.append(
            {
                "role": "user",
                "content": "Describe this image:",
                "images": [img]  # Replace with the actual image path
            }
        )

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
        if 'message' in chunk and 'content' in chunk['message']:
            buffer += chunk['message']['content']
            yield buffer


def format_history(history, message):
    if message["text"] is not None:
        history.append(ChatMessage(role="user", content=message["text"]))
    elif len(message["files"]) >= 1:
        history.append(ChatMessage(role="user", content="explain this"))
    message_history = []
    img_list = []
    for single_msg in history:
        if type(single_msg) is ChatMessage:  # ignoring the path in message_history for ollama
            message_history.append({
                "role": single_msg.role,
                "content": single_msg.content
            })
    for file in message["files"]:
        history.append(ChatMessage(role="user", content=FileData(path=file)))
        img_b64 = image_to_base64(file)
        img_list.append(f"{img_b64}")
        message_history[-1]["images"] = img_list
    return message_history

with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
              footer {visibility: hidden}
              #warning {background-color: green; text-align: center;}
              """,
        fill_height=True,
) as io:
    chatbot = gr.Chatbot(
        label="Chatbot",
        scale=1,
        type="messages",
        autoscroll=True,
    )

    with gr.Sidebar():
        gr.Markdown("""
        Image Input
        """)
        image = gr.Image(type="filepath")

    gr.Markdown("""
    # Unclassified
    """,
                elem_id="warning", )

    gr.ChatInterface(
        stream_chat,
        type="messages",
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            image,
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.8,
                label="Temperature",
                render=False,
                info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
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
                info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
                render=False,
            ),
            gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=20,
                label="top_k",
                info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
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
            gr.Dropdown(
                choices=model_names,
                filterable=False,
                label="Model",
                render=False,
            ),
        ],
    )

app = gr.mount_gradio_app(app, io, path="")

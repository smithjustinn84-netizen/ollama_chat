from fastapi import FastAPI
import gradio as gr
import ollama

# Create the server
app = FastAPI()

# Create a client to talk to Ollama
client = ollama.Client()
# get a list of available models for the LLM
model_list = ollama.list()
model_names = [model['model'] for model in model_list['models']]


def stream_chat(prompt: str | None, history: list, images: list[str], temperature: float, max_new_tokens: int,
                top_p: float, top_k: int,
                penalty: float, model: str = 'llava'):
    """
    Stream a llm response for the given message with the provided params and models to a gradio chatbot.

    Inputs: [prompt, history, images, temperature, max_new_tokens, top_p, penalty, model]
    Outputs: [history]
    """

    # add messages to history
    history.append({
        "role": "user",
        "content": prompt,
    })

    # convert history to conversation format to send to the LLM.
    # The LLM expects a list of messages in the format:
    # [{"role": "system", "content": "", "images": [path.jpeg,]}, {"role": "user", "content": "What is the meaning of life?"}]
    conversation = [
        {"role": "system", "content": "You are a professional assistant."},
    ]

    for message in history:
        content = message["content"]
        if isinstance(content, dict) and "content" in content:
            conversation.append({
                "role": "user",
                "content": "",
                "images": [content["path"]],
            })
        elif "content" in content:
            conversation.append({
                "role": "user",
                "content": content["content"],
            })

    if len(images) > 0:
        conversation = [*history, {"role": "user", "content": "", "images": images}]
    else:
        conversation = [*history, {"role": "user", "content": prompt}]

    # Send the conversation off to the LLM with the selected model and params
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


def image_input():
    # image input UI
    gr.Markdown("""
        Image Input
        """)
    image = gr.Image(type="filepath")

    def add_images(img: str, save_images: list[str]):
        if not img:
            raise gr.Error("No Image to add")
        return save_images + [img], gr.Image(value=None)

    gr.Button("Save Image").click(fn=add_images, inputs=[image, saved_images], outputs=[saved_images, image])
    gr.ClearButton(value="Clear Saved", components=[saved_images])

    @gr.render(inputs=saved_images)
    def show_images(imgs: list[str]):
        if len(imgs) == 0:
            gr.Markdown("## No Saved Images")
        else:
            for img in imgs:
                gr.Image(type="filepath", value=img)


# Gradio Chatbot UI https://www.gradio.app/docs/gradio/chatbot
with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
              footer {visibility: hidden}
              #warning {background-color: green; text-align: center;}
              """,
        fill_height=True,
) as io:
    saved_images = gr.State([])  # Images saved by the user to ask the LLM about
    chatbot = gr.Chatbot(
        label="Chatbot",
        scale=1,
        type="messages",
        autoscroll=True,
    )

    # Sidebar for image input
    with gr.Sidebar():
        image_input()

    gr.Markdown("""
    # Unclassified
    """,
                elem_id="warning", )

    gr.ChatInterface(
        stream_chat,
        type="messages",
        chatbot=chatbot,
        textbox=gr.Textbox(
            interactive=True,
            placeholder="Enter message",
            show_label=False,
        ),
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            saved_images,
            # LLM Params
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

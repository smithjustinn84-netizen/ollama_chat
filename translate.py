import gradio as gr
import ollama


def translate_text(text, target_language):
    language_map = {
        "Turkish": "Turkish",
        "Spanish": "Spanish",
        "Chinese": "Chinese (Simplified)",
    }
    prompt = f"Translate the following English text to {language_map[target_language]}:\n\nEnglish: {text}\n\n{target_language} translation:"
    try:
        response = ollama.chat(
            model='llama2',
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt},
            ])
        print(response['message']['content'])
        return response.message.content
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
   fn=translate_text,
   inputs=[
       gr.Textbox(
           lines=4,
           placeholder="Enter English text to translate...",
           label="English Text",
       ),
       gr.Dropdown(choices=["Turkish", "Spanish", "Chinese"], label="Target Language"),
   ],
   outputs=gr.Textbox(label="Translation", show_copy_button=True),
   title="English to Turkish/Spanish/Chinese Translator",
   description="Translate English text to Turkish, Spanish, or Chinese. Enter the text you want to translate, and select the target language.",
)
iface.launch()
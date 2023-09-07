import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", max_length=1000,
                     model="EasthShin/Youth_Chatbot_Kogpt2-base")


def chat(msg, history):
    prompt = '\n'.join([f"유저: {h[0]}\n챗봇: {h[1]}" for h in history[-3:]])
    prompt = f"{prompt}\n유저: {msg}\n챗봇: "
    gen_text = generator(prompt)[0]['generated_text']
    history.append((msg, gen_text[len(prompt):]))

    return "", history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

demo.launch(debug=True)

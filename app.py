import os
import gradio as gr
from dotenv import load_dotenv
from rag_pipeline import build_rag_chain, format_answer

load_dotenv()
assert os.getenv("GOOGLE_API_KEY")

qa = build_rag_chain()

def answer(query):
    if not query or not query.strip():
        return "Please enter a question."
    result = qa.invoke(query)   
    return format_answer(result)

with gr.Blocks(title="Gemini RAG with LangChain") as demo:
    gr.Markdown("# Gemini RAG with LangChain")
    gr.Markdown("Ask questions grounded in your local documents (./data).")

    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g., What does my PDF say about topic X?"
    )
    out = gr.Markdown(label="Answer")

    btn = gr.Button("Ask")
    btn.click(fn=answer, inputs=inp, outputs=out)

    # Allow pressing Enter to submit
    inp.submit(fn=answer, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

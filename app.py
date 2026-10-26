import shutil
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from ingest import KNOWLEDGE_BASE_PATH, list_books, fetch_selected_document, fetch_uploaded_document
from chunk import create_chunks, STRATEGIES
from encoding import create_embeddings
from auto_regressive_llm import answer_question, get_collection
from providers import EMBEDDING_MODELS, CHAT_MODELS, embeddings_for
from eval import build_eval_tab

load_dotenv(override=True)


def book_is_embedded(book_name, emb_model):
    collection = get_collection(emb_model)
    existing = collection.get(where={"name": book_name}, limit=1)
    return len(existing["ids"]) > 0


def ingest_book(document, strategy, chunk_size, chunk_overlap, embeddings, emb_model):
    book_name = document["name"]
    if book_is_embedded(book_name, emb_model):
        print(f"'{book_name}' is already embedded with {emb_model}, skipping chunk+embed")
        return book_name

    chunks = create_chunks([document], strategy, chunk_size, chunk_overlap, embeddings)
    create_embeddings(chunks, embeddings, emb_model)
    return book_name


def book_label(book_name):
    return f"📖 Currently chatting about: **{book_name}**"


def select_book(pdf_name, strategy, chunk_size, chunk_overlap, chat_model, embedding_model):
    if pdf_name is None:
        return None, "*No book selected yet*", "", []

    embeddings = embeddings_for(embedding_model)
    document = fetch_selected_document(pdf_name, chat_model)
    book_name = ingest_book(document, strategy, chunk_size, chunk_overlap, embeddings, embedding_model)

    return book_name, book_label(book_name), "", []


def upload_book(file_path, strategy, chunk_size, chunk_overlap, chat_model, embedding_model):
    embeddings = embeddings_for(embedding_model)

    try:
        document = fetch_uploaded_document(file_path, chat_model)
    except ValueError as e:
        raise gr.Error(str(e))

    destination = KNOWLEDGE_BASE_PATH / Path(file_path).name
    if not destination.exists():
        shutil.copy(file_path, destination)

    book_name = ingest_book(document, strategy, chunk_size, chunk_overlap, embeddings, embedding_model)

    return gr.Dropdown(choices=list_books(), value=destination.name), book_name, book_label(book_name), "", []


def format_context(question, chunks):
    block = f"### Q: {question}\n\n"
    for chunk in chunks:
        block += f"**Source:** {chunk.metadata['source']}\n\n{chunk.page_content}\n\n---\n\n"
    return block


def chat(history, book_name, context_markdown, embedding_model, chat_model):
    if book_name is None:
        raise gr.Error("Please select or upload a book first")

    question = history[-1]["content"]
    prior = history[:-1]

    embeddings = embeddings_for(embedding_model)

    answer, chunks = answer_question(question, book_name, embeddings, embedding_model, chat_model, prior)
    history.append({"role": "assistant", "content": answer})

    return history, context_markdown + format_context(question, chunks)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Talking Books", theme=theme) as ui:
        with gr.Tabs():
            with gr.Tab("💬 Chat"):
                gr.Markdown("# 📚 Talking Books\nAsk me anything about a book!")

                book_name_state = gr.State(None)

                with gr.Row():
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(
                            label="💬 Conversation", height=500, type="messages", show_copy_button=True
                        )
                        message = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about the book...",
                            show_label=False,
                        )

                        gr.Markdown("### 📖 Book")
                        current_book_label = gr.Markdown("*No book selected yet*")
                        with gr.Row():
                            book_dropdown = gr.Dropdown(choices=list_books(), value=None, label="Select a book")
                            file_upload = gr.File(
                                label="Or drop in a new PDF", file_types=[".pdf"], file_count="single", type="filepath"
                            )

                        gr.Markdown("### ⚙️ Models")
                        with gr.Row():
                            chunk_strategy = gr.Dropdown(choices=STRATEGIES, value=STRATEGIES[0], label="Chunking Strategy")
                            embedding_model = gr.Dropdown(choices=EMBEDDING_MODELS, value=EMBEDDING_MODELS[0], label="Embedding Model")
                            chat_model = gr.Dropdown(choices=CHAT_MODELS, value=CHAT_MODELS[0], label="Chat Model")

                        with gr.Row():
                            chunk_size = gr.Slider(500, 4000, value=2000, step=100, label="Chunk Size")
                            chunk_overlap = gr.Slider(0, 500, value=200, step=50, label="Chunk Overlap")

                    with gr.Column(scale=1):
                        context_markdown = gr.Markdown(
                            label="📚 Retrieved Context",
                            value="*Retrieved context will appear here*",
                            container=True,
                            height=700,
                        )

                book_dropdown.change(
                    select_book,
                    inputs=[book_dropdown, chunk_strategy, chunk_size, chunk_overlap, chat_model, embedding_model],
                    outputs=[book_name_state, current_book_label, context_markdown, chatbot],
                )

                embedding_model.change(
                    select_book,
                    inputs=[book_dropdown, chunk_strategy, chunk_size, chunk_overlap, chat_model, embedding_model],
                    outputs=[book_name_state, current_book_label, context_markdown, chatbot],
                )

                file_upload.upload(
                    upload_book,
                    inputs=[file_upload, chunk_strategy, chunk_size, chunk_overlap, chat_model, embedding_model],
                    outputs=[book_dropdown, book_name_state, current_book_label, context_markdown, chatbot],
                )

                message.submit(
                    put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
                ).then(
                    chat,
                    inputs=[chatbot, book_name_state, context_markdown, embedding_model, chat_model],
                    outputs=[chatbot, context_markdown],
                )

            with gr.Tab("📊 Evaluations"):
                build_eval_tab()

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()

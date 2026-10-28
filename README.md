# talking-books

![Fox reading a book](assets/fox-reading.png)

Using Retrieval-Augmented Generation (RAG) to "chat" with my favorite books.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Run

```bash
python app.py
```

This launches the Gradio app in your browser. Pick a book from the dropdown (or upload your own PDF), wait for it to ingest, then start chatting. The Evaluations tab lets you score retrieval and answer quality against the test sets in `testing/jsonl_tests/`.

#injest.py

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt

from providers import chat

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "books_data"

wait = wait_exponential(multiplier=1, min=10, max=240)
stop = stop_after_attempt(3)

class Title(BaseModel):
    title: str = Field(description="The title of the book, exactly as it appears on the cover or title page")


def pdf_to_markdown(file):
    pdf_loader = PyPDFLoader(str(file))
    pages = pdf_loader.load()
    return "\n\n".join(f"## Page {i + 1}\n\n{page.page_content}" for i, page in enumerate(pages))


@retry(wait=wait, stop=stop)
def find_title(markdown, model):
    first_pages = markdown[:3000]  # title is usually on the first few pages
    messages = [{"role": "user", "content": f"What is the title of this book? Reply with just the title, nothing else.\n\n{first_pages}"}]
    response = chat(model, messages, response_format=Title)
    reply = response.choices[0].message.content
    return Title.model_validate_json(reply).title


def book_id(file):
    """Stable identifier for a book, independent of which chat model is asked for its title."""
    return Path(file).stem


def fetch_uploaded_document(file_path, model):
    """For when a user drops in their own pdf."""
    file = Path(file_path)
    if file.suffix.lower() != ".pdf":
        raise ValueError("Please upload a PDF file")

    markdown = pdf_to_markdown(file)
    name = find_title(markdown, model)
    return {"id": book_id(file), "source": file.as_posix(), "name": name, "text": markdown}


def fetch_selected_document(book_name, model):
    """For when a user picks a book from the dropdown of books already in the knowledge base."""
    file = KNOWLEDGE_BASE_PATH / book_name
    markdown = pdf_to_markdown(file)
    name = find_title(markdown, model)
    return {"id": book_id(file), "source": file.as_posix(), "name": name, "text": markdown}


def list_books():
    return [file.name for file in KNOWLEDGE_BASE_PATH.glob("*.pdf")]

import re

from pydantic import BaseModel
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
)

from ingest import fetch_selected_document, list_books
from encoding import create_embeddings
from auto_regressive_llm import get_collection

STRATEGIES = ["recursive", "markdown", "semantic", "token", "chapter", "hierarchical", "sliding_window"]

CHAPTER_PATTERN = r"(?im)^\s*(chapter|part)\s+\S+\s*$"


def split_text(text, strategy, chunk_size, chunk_overlap, embeddings):
    """Returns a list of (page_content, extra_metadata) pairs."""
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return [(chunk, {}) for chunk in splitter.split_text(text)]

    if strategy == "markdown":
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
        return [(doc.page_content, {}) for doc in splitter.split_text(text)]

    if strategy == "token":
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return [(chunk, {}) for chunk in splitter.split_text(text)]

    if strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(embeddings)
        return [(chunk, {}) for chunk in splitter.split_text(text)]

    if strategy == "sliding_window":
        splitter = CharacterTextSplitter(separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return [(chunk, {}) for chunk in splitter.split_text(text)]

    if strategy == "chapter":
        # Splits on lines like "Chapter 1" / "CHAPTER ONE" / "Part II" - falls back to one block if none match
        splitter = CharacterTextSplitter(
            separator=CHAPTER_PATTERN,
            is_separator_regex=True,
            keep_separator=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return [(chunk, {}) for chunk in splitter.split_text(text)]

    if strategy == "hierarchical":
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size * 4, chunk_overlap=0)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pairs = []
        for parent_text in parent_splitter.split_text(text):
            for child_text in child_splitter.split_text(parent_text):
                pairs.append((child_text, {"parent_content": parent_text}))
        return pairs

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


class Result(BaseModel):
    page_content: str
    metadata: dict


def chapter_offsets(text):
    return [match.start() for match in re.finditer(CHAPTER_PATTERN, text)]


def chapters_for_chunk(offsets, start, end):
    """Chapter numbers (1-indexed) whose heading is at or before `start`, or falls within [start, end)."""
    if start is None or not offsets:
        return []
    chapters = [i + 1 for i, offset in enumerate(offsets) if offset <= start]
    chapters = chapters[-1:]
    chapters += [i + 1 for i, offset in enumerate(offsets) if start < offset < end]
    return chapters


def create_chunks(documents, strategy, chunk_size, chunk_overlap, embeddings=None):
    """Split each document with the selected splitter"""
    results = []
    for document in documents:
        text = document["text"]
        offsets = chapter_offsets(text)
        cursor = 0
        for i, (raw_text, extra_metadata) in enumerate(split_text(text, strategy, chunk_size, chunk_overlap, embeddings)):
            start = text.find(raw_text, cursor)
            if start == -1:
                start = text.find(raw_text)
            cursor = start + len(raw_text) if start != -1 else cursor
            chapters = chapters_for_chunk(offsets, start, cursor)

            metadata = {
                "id": document["id"],
                "source": document["source"],
                "name": document["name"],
                "chunk_index": i,
                "chapters": ",".join(str(c) for c in chapters),  # Chroma metadata must be flat scalars, not lists
                **extra_metadata,
            }
            results.append(Result(page_content=raw_text, metadata=metadata))

    print(f"Divided into {len(results)} chunks using '{strategy}'")
    return results


def book_already_exists(book_id, emb_model):
    collection = get_collection(emb_model)
    existing = collection.get(where={"id": book_id}, limit=1)
    return len(existing["ids"]) > 0


def ingest_book(document, strategy, chunk_size, chunk_overlap, embeddings, emb_model):
    if book_already_exists(document["id"], emb_model):
        print(f"'{document['name']}' is already embedded with {emb_model}, skipping chunk+embed")
        return document

    chunks = create_chunks([document], strategy, chunk_size, chunk_overlap, embeddings)
    create_embeddings(chunks, embeddings, emb_model)
    return document


# if __name__ == "__main__":
#     model = "openai/gpt-4.1-nano"
#     chunk_size = 2000
#     chunk_overlap = 200
#     documents = [fetch_selected_document(book_name, model) for book_name in list_books()]
#     chunks = create_chunks(documents, "recursive", chunk_size, chunk_overlap)

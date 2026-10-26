from pydantic import BaseModel
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
)

from ingest import fetch_selected_document, list_books

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
        # CharacterTextSplitter with no multi-separator fallback - a literal fixed-size overlapping window
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
        # langchain has no plain splitter for this, so this is a manual parent/child pass:
        # big parent sections, each re-split into normal-sized child chunks that remember their parent.
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size * 4, chunk_overlap=0)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pairs = []
        for parent_text in parent_splitter.split_text(text):
            for child_text in child_splitter.split_text(parent_text):
                pairs.append((child_text, {"parent_content": parent_text}))
        return pairs

    raise ValueError(f"Unknown chunking strategy: {strategy}")


class Result(BaseModel):
    page_content: str
    metadata: dict


def create_chunks(documents, strategy, chunk_size, chunk_overlap, embeddings=None):
    """Split each document with the selected splitter. No AI calls - these are all free,
    local splitters (semantic uses the embedding model to find breakpoints, not a chat model)."""
    results = []
    for document in documents:
        for raw_text, extra_metadata in split_text(document["text"], strategy, chunk_size, chunk_overlap, embeddings):
            metadata = {"source": document["source"], "name": document["name"], **extra_metadata}
            results.append(Result(page_content=raw_text, metadata=metadata))

    print(f"Divided into {len(results)} chunks using '{strategy}'")
    return results


if __name__ == "__main__":
    model = "openai/gpt-4.1-nano"
    chunk_size = 2000
    chunk_overlap = 200
    documents = [fetch_selected_document(book_name, model) for book_name in list_books()]
    chunks = create_chunks(documents, "recursive", chunk_size, chunk_overlap)

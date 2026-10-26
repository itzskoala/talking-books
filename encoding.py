from pathlib import Path
from chromadb import PersistentClient
from litellm import embedding

DB_NAME = str(Path(__file__).parent / "preprocessed_db")


class LiteLLMEmbeddings:
    """Wraps litellm.embedding so any provider litellm supports can be used as an embedding model."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        response = embedding(model=self.model, input=texts)
        return [item["embedding"] for item in response.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def create_embeddings(chunks, embeddings, emb_model):
    chroma = PersistentClient(path=DB_NAME)

    collection_name = f"books_{emb_model}".replace("/", "_")
    collection = chroma.get_or_create_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    ids = [f"{chunk.metadata['source']}-{i}" for i, chunk in enumerate(chunks)]
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
    )

    print(f"Added {len(chunks)} chunks to '{collection_name}'")
    print(f"Vectorstore now has {collection.count()} documents")

    return collection_name

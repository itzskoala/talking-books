from pathlib import Path
from chromadb import PersistentClient
from litellm import embedding
from tenacity import retry, wait_exponential, stop_after_attempt

DB_NAME = str(Path(__file__).parent / "preprocessed_db")

wait = wait_exponential(multiplier=1, min=10, max=240)
stop = stop_after_attempt(3)


class LiteLLMEmbeddings:
    def __init__(self, model):
        self.model = model

    @retry(wait=wait, stop=stop)
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

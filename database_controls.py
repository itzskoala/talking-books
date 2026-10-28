from chromadb import PersistentClient

from encoding import DB_NAME
from auto_regressive_llm import get_collection


def clear_db(emb_model, book_id):
    collection_name = f"books_{emb_model}".replace("/", "_")
    client = PersistentClient(path=DB_NAME)
    if collection_name not in [c.name for c in client.list_collections()]:
        return

    collection = client.get_collection(collection_name)
    matches = collection.get(where={"id": book_id}, limit=1)
    if not matches["ids"]:
        return

    collection.delete(where={"id": book_id})
    print(f"Deleted chunks for '{book_id}' from '{collection_name}'")


def print_book_vectors(book_id, emb_model):
    collection = get_collection(emb_model)
    results = collection.get(where={"id": book_id})
    print(f"{len(results['ids'])} chunks for '{book_id}' in '{collection.name}'")
    for id, doc in zip(results["ids"], results["documents"]):
        print(f"- {id}: {doc[:150]}...")

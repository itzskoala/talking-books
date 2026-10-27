from chromadb import PersistentClient

from encoding import DB_NAME
from auto_regressive_llm import get_collection


def clear_db(emb_model):
    collection = get_collection(emb_model)
    PersistentClient(path=DB_NAME).delete_collection(collection.name)
    print(f"Deleted collection '{collection.name}'")


def print_book_vectors(book_name, emb_model):
    collection = get_collection(emb_model)
    results = collection.get(where={"name": book_name})
    print(f"{len(results['ids'])} chunks for '{book_name}' in '{collection.name}'")
    for id, doc in zip(results["ids"], results["documents"]):
        print(f"- {id}: {doc[:150]}...")

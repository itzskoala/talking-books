import sys

from providers import embeddings_for, EMBEDDING_MODELS, CHAT_MODELS
from auto_regressive_llm import answer_question
from testing.evaluations import resolve_book_name, evaluate_retrieval, evaluate_answer
from testing.test import load_tests


def chat_with_book(question, book_name, embeddings, emb_model, chat_model):
    """Ask the RAG pipeline one question directly (no jsonl test involved) and print the result."""
    answer, chunks = answer_question(question, book_name, embeddings, emb_model, chat_model)

    print(f"\n{'=' * 80}")
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"\nRetrieved {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"- {chunk.metadata['source']}: {chunk.page_content[:150]}...")
    print(f"{'=' * 80}\n")


#this will need to be combined with the main gradio app, so two pages that are apart of one app, the evaluations doesn't load obviously until the user starts asking questions and the app runs/chunks/returns/ect...

def run_cli_evaluation(test_set, embeddings, emb_model, model, test_number):
    """Run evaluation for a specific test (CLI helper)."""
    print(f"\n{'=' * 80}")
    print(f"Test set: {test_set}")
    print(f"Embedding model: {emb_model}")
    print(f"Autoregressive model: {model}")
    print(f"{'=' * 80}")

    book_name = resolve_book_name(test_set, model)
    tests = load_tests(test_set)
    print(f"Loaded {len(tests)} tests from '{test_set}.jsonl'")

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")
    print("Fetching context (rewrite + unadjusted/adjusted retrieval + merge + rerank)...")

    retrieval_result = evaluate_retrieval(test, book_name, embeddings, emb_model, model)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")
    print("Generating answer, then asking the LLM judge to score it...")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test, book_name, embeddings, emb_model, model)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    # testing/jsonl_tests/<test_set>.jsonl - which book's test questions to use
    test_set = "harry_potter_1"
    # row number inside that jsonl file - which specific question to evaluate
    test_number = 0

    # EMBEDDING_MODELS/CHAT_MODELS are defined in providers.py, all routed through OpenRouter
    # except ollama (that one's Ollama Cloud directly). CHAT_MODELS order: 0 = gemini,
    # 1 = chatgpt/openai, 2 = ollama cloud, 3 = huggingface (open-weight). Only one
    # embedding model is listed - OpenRouter doesn't support embeddings for the others.
    emb_model = EMBEDDING_MODELS[0]
    chat_model = CHAT_MODELS[0]
    embeddings = embeddings_for(emb_model)

    # resolve once so both the manual chat below and the evaluation below use the same book
    book_name = resolve_book_name(test_set, chat_model)

    # 1. chat with the book directly and see the raw answer + context
    chat_with_book("What is Harry's numerical IQ score?", book_name, embeddings, emb_model, chat_model)

    # 2. run the formal test-set evaluation (retrieval metrics + LLM-judged answer score)
    run_cli_evaluation(test_set, embeddings, emb_model, chat_model, test_number)

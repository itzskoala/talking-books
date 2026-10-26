import sys
import math
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from test import load_tests, pdf_for_test_set
from auto_regressive_llm import answer_question, fetch_context
from ingest import fetch_selected_document
from providers import chat, CHAT_MODELS


load_dotenv(override=True)

JUDGE_MODEL = CHAT_MODELS[1]  # a fixed judge model, independent of whichever model is under test


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


def calculate_mrr(keyword, retrieved_docs):
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances, k):
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword, retrieved_docs, k=10):
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]

    dcg = calculate_dcg(relevances, k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def resolve_book_name(test_set, model):
    """
    The jsonl test set name (e.g. "harry_potter_1") is just a human-picked slug - it
    has no reliable relationship to the book's real title, which is what chroma's
    metadata is actually filtered on. So look up which pdf this test set covers, then
    run the same AI title-extraction call ingest.py uses, to get the real book_name.
    """
    pdf_name = pdf_for_test_set(test_set)
    print(f"Test set '{test_set}' -> pdf '{pdf_name}' -> asking AI for the book title...")
    document = fetch_selected_document(pdf_name, model)
    print(f"Resolved book title: '{document['name']}'")
    return document["name"]


def evaluate_retrieval(test, book_name, embeddings, emb_model, model, k=10):
    """
    Evaluate retrieval performance for a test question, using the same
    fetch_context pipeline (rewrite + merge + rerank) the chat uses.
    """
    retrieved_docs = fetch_context(test.question, book_name, embeddings, emb_model, model)

    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(test, book_name, embeddings, emb_model, model):
    """
    Evaluate answer quality using LLM-as-a-judge.

    Returns a tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    generated_answer, retrieved_docs = answer_question(test.question, book_name, embeddings, emb_model, model)

    # LLM judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    judge_response = chat(JUDGE_MODEL, judge_messages, response_format=AnswerEval)
    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)

    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval(test_set, embeddings, emb_model, model):
    """Evaluate all retrieval tests for a test set."""
    book_name = resolve_book_name(test_set, model)
    tests = load_tests(test_set)
    total_tests = len(tests)
    print(f"Running retrieval evaluation on {total_tests} tests for '{book_name}'...")
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test, book_name, embeddings, emb_model, model)
        progress = (index + 1) / total_tests
        print(f"  [{index + 1}/{total_tests}] mrr={result.mrr:.2f} ndcg={result.ndcg:.2f} coverage={result.keyword_coverage:.0f}% - {test.question}")
        yield test, result, progress


def evaluate_all_answers(test_set, embeddings, emb_model, model):
    """Evaluate all answers to tests for a test set."""
    book_name = resolve_book_name(test_set, model)
    tests = load_tests(test_set)
    total_tests = len(tests)
    print(f"Running answer evaluation on {total_tests} tests for '{book_name}'...")
    for index, test in enumerate(tests):
        result = evaluate_answer(test, book_name, embeddings, emb_model, model)[0]
        progress = (index + 1) / total_tests
        print(f"  [{index + 1}/{total_tests}] accuracy={result.accuracy:.1f} completeness={result.completeness:.1f} relevance={result.relevance:.1f} - {test.question}")
        yield test, result, progress





import json
from pathlib import Path
from pydantic import BaseModel, Field

TESTS_DIR = Path(__file__).parent / "jsonl_tests"

BOOK_PDF_MAP = {
    "declaration_of_indepedence": "declaration_of_indepedence.pdf",
    "dexter": "Darkly-Dreaming-Dexter.pdf",
    "harry_potter_1": "harrypotter_1.pdf",
    "time_travelers_wife": "Time-Travelers-Wife-EBook.pdf",
}


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    source_chapter: str | int | list[int] | list[str] | None = Field(description="Which chapter(s) the context was found in")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests(test_set):
    """Load test questions from testing/jsonl_tests/<test_set>.jsonl."""
    test_file = TESTS_DIR / f"{test_set}.jsonl"
    tests = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            tests.append(TestQuestion(**data))
    return tests


def pdf_for_test_set(test_set):
    return BOOK_PDF_MAP[test_set]


def list_test_sets():
    return [file.stem for file in TESTS_DIR.glob("*.jsonl")]

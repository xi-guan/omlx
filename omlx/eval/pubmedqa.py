# SPDX-License-Identifier: Apache-2.0
"""PubMedQA benchmark (expert-labeled subset).

Tests reading comprehension over biomedical literature: given a PubMed
abstract and its title question, answer yes / no / maybe.
Dataset bundled from qiaojin/PubMedQA (pqa_labeled) on HuggingFace.
1,000 expert-annotated questions.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import load_jsonl, stratified_sample

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
VALID_ANSWERS = ("yes", "no", "maybe")


def _extract_decision(response: str) -> str:
    """Extract yes/no/maybe from a response, preferring an explicit answer."""
    text = response.strip().lower()
    # explicit "answer is X" / "answer: X" wins so reasoning drafts don't leak
    matches = re.findall(r"answer\s*(?:is|:)\s*\**\s*(yes|no|maybe)\b", text)
    if matches:
        return matches[-1]
    matches = re.findall(r"\b(yes|no|maybe)\b", text)
    return matches[-1] if matches else ""


class PubMedQABenchmark(BaseBenchmark):
    """PubMedQA: 0-shot yes/no/maybe answering over a given abstract."""

    name = "pubmedqa"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load PubMedQA from bundled data."""
        raw_items = load_jsonl(DATA_DIR / "pubmedqa_labeled.jsonl")

        items = []
        for raw in raw_items:
            if raw.get("answer") not in VALID_ANSWERS or not raw.get("contexts"):
                continue
            items.append({
                "id": raw["id"],
                "question": raw["question"],
                "contexts": raw["contexts"],
                "answer": raw["answer"],
            })

        logger.info(f"PubMedQA: loaded {len(items)} questions")

        self.dataset_total = len(items)
        if sample_size == 0:
            return items

        # keep the yes/no/maybe ratio so the rare "maybe" class survives sampling
        return stratified_sample(items, sample_size, key="answer")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as abstract + question, answered with one word."""
        abstract = "\n".join(item["contexts"])

        parts = [
            "Answer the question based on the abstract. "
            "Answer with just one word: yes, no, or maybe.\n",
            f"Abstract: {abstract}\n",
            f"Question: {item['question']}\n",
            "Answer:",
        ]

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return _extract_decision(response)

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> Optional[str]:
        # gold label as category: exposes yes-bias and "maybe" accuracy
        return item.get("answer")

# SPDX-License-Identifier: Apache-2.0
"""MedQA (USMLE) benchmark.

Tests clinical knowledge with USMLE-style 4-option multiple choice.
Dataset bundled from GBaker/MedQA-USMLE-4-options (test split) on HuggingFace.
1,273 questions tagged step1 or step2&3.
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import load_jsonl, stratified_sample

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class MedQABenchmark(BaseBenchmark):
    """MedQA: 0-shot USMLE multiple choice (A-D)."""

    name = "medqa"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MedQA from bundled data."""
        raw_items = load_jsonl(DATA_DIR / "medqa_usmle_test.jsonl")

        items = []
        for raw in raw_items:
            options = raw.get("options", {})
            if not options or raw.get("answer") not in options:
                continue
            items.append({
                "id": raw["id"],
                "question": raw["question"],
                "options": options,
                "answer": raw["answer"],
                "meta_info": raw.get("meta_info", "unknown"),
            })

        logger.info(f"MedQA: loaded {len(items)} questions")

        self.dataset_total = len(items)
        if sample_size == 0:
            return items

        return stratified_sample(items, sample_size, key="meta_info")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as multiple choice with lettered options."""
        parts = [
            "Answer the following USMLE-style medical question. "
            "Answer with just the letter.\n",
            f"Question: {item['question']}\n",
        ]
        for label in sorted(item["options"]):
            parts.append(f"{label}. {item['options'][label]}")

        parts.append("\nAnswer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, sorted(item["options"]))

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> Optional[str]:
        return item.get("meta_info")

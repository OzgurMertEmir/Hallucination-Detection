"""Dataset wrapper for the HaluEval benchmark.

This module normalizes the `qa`, `dialogue`, and `summarization` splits of
the dataset into a shared QAExample structure for downstream processing.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import logging

from .examples import QAExample

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore

logger = logging.getLogger(__name__)


class HaluEvalDataset:
    """Loads and serves examples from the HaluEval dataset.

    Parameters
    ----------
    config : str
        Which configuration of HaluEval to load. Must be one of `dialogue`,
        `general`, `qa`, `summarization` or the corresponding *_samples
        variants. Normalization to QAExample is implemented for `qa`,
        `dialogue`, and `summarization`.
    split : str
        Which split to load. The HaluEval dataset provides a single split
        `data` for all configurations.
    """

    def __init__(self, config: str = "qa", split: str = "data") -> None:
        logger.info("Loading HaluEval dataset...")
        self.config = config
        self.split = split
        self.data: Optional[Iterable[Dict]] = None
        self._load()

    def _load(self) -> None:
        if load_dataset is None:
            raise ImportError(
                "datasets library is not available. Please install datasets to use HaluEvalDataset."
            )
        try:
            self.data = load_dataset("pminervini/HaluEval", self.config, split=self.split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HaluEval configuration '{self.config}'. "
                "Ensure you have an internet connection and the `datasets` package installed."
            ) from exc

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    def examples(self, max_examples: Optional[int] = None) -> Iterable[QAExample]:
        """Iterate over examples in the dataset.

        Parameters
        ----------
        max_examples : int, optional
            If provided, stop iteration after this many examples.
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded; call _load() before iterating.")

        count = 0
        for idx, row in enumerate(self.data):
            ex = self._normalize(row, idx)
            yield ex
            count += 1
            if max_examples is not None and count >= max_examples:
                return

    def _normalize(self, row: Dict, idx: int) -> QAExample:
        """Convert a raw dataset row into a QAExample."""
        cfg = (self.config or "").lower()
        qid = f"halueval/{cfg}:{idx}"

        if cfg == "qa":
            return QAExample(
                qid=qid,
                knowledge=str(row.get("knowledge", "") or ""),
                question=str(row.get("question", "") or ""),
                right_answer=str(row.get("right_answer", "") or ""),
                hallucinated_answer=str(row.get("hallucinated_answer", "") or ""),
            )
        if cfg == "dialogue":
            return QAExample(
                qid=qid,
                knowledge=str(row.get("knowledge", "") or ""),
                question=str(row.get("dialogue_history", "") or ""),
                right_answer=str(row.get("right_response", "") or ""),
                hallucinated_answer=str(row.get("hallucinated_response", "") or ""),
            )
        if cfg == "summarization":
            return QAExample(
                qid=qid,
                knowledge=str(row.get("document", "") or ""),
                question="Please summarize the passage.",
                right_answer=str(row.get("right_summary", "") or ""),
                hallucinated_answer=str(row.get("hallucinated_summary", "") or ""),
            )

        raise ValueError(f"Normalization for HaluEval config '{self.config}' is not implemented.")

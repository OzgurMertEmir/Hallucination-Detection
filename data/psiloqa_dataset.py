"""
psiloqa_dataset
===============

Wrapper for the multilingual PsiloQA dataset hosted on the Hugging Face Hub.

PsiloQA pairs Wikipedia passages with questions, golden answers and model
hypotheses that may contain hallucinations. This wrapper normalizes each row
into a :class:`~hallucination_detector.data.examples.QAExample`, treating the
retrieved passage as knowledge, the golden answer as the ground truth, and the
LLM hypothesis as a potential hallucinated answer.

Example
-------

>>> from hallucination_detector.data.psiloqa_dataset import PsiloQADataset
>>> ds = PsiloQADataset(split="validation", languages=["en"])
>>> ex = next(ds.examples())
>>> ex.question
'Which ocean is Palmyra Atoll located in?'
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence, Set

from data.examples import QAExample

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore

logger = logging.getLogger(__name__)


class PsiloQADataset:
    """Load and iterate over the PsiloQA dataset.

    Parameters
    ----------
    split : str, optional
        Dataset split to load. Valid values include ``"train"``,
        ``"validation"`` and ``"test"``. Defaults to ``"train"``.
    languages : Sequence[str], optional
        Optional list of ISO language codes to retain (case‑insensitive).
        When provided, only rows whose ``lang`` field matches are returned.
    dataset_name : str, optional
        Identifier passed to :func:`datasets.load_dataset`. Override this if
        using a local copy of the dataset.
    """

    DEFAULT_DATASET = "s-nlp/PsiloQA"

    def __init__(
        self,
        split: str = "train",
        languages: Optional[Sequence[str]] = ['en'],
        dataset_name: str = DEFAULT_DATASET,
    ) -> None:
        self.split = split
        self.dataset_name = dataset_name
        self.languages: Optional[Set[str]] = {lang.lower() for lang in languages} if languages else None
        self.data: Optional[Iterable[Dict]] = None
        self._length: int = 0
        self._load()

    def _load(self) -> None:
        if load_dataset is None:
            raise ImportError(
                "datasets library is not available. Please install datasets to use PsiloQADataset."
            )
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PsiloQA split '{self.split}'. Ensure you have an internet connection "
                f"and that the `datasets` package is installed."
            ) from exc

        # Optionally filter by language
        if self.languages:
            lang_set = self.languages
            dataset = dataset.filter(lambda row: (row.get("lang") or "").lower() in lang_set)
        self.data = dataset
        self._length = len(dataset)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def examples(self, max_examples: Optional[int] = None) -> Iterable[QAExample]:
        """Yield examples from the dataset as :class:`QAExample` objects.

        Parameters
        ----------
        max_examples : int, optional
            Optionally limit the number of examples produced. Useful when
            sampling large datasets.
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded; call _load() before iterating.")

        count = 0
        for idx, row in enumerate(self.data):
            ex = self._normalize(row, idx)
            if ex is None:
                continue
            yield ex
            count += 1
            if max_examples is not None and count >= max_examples:
                break

    def _normalize(self, row: Dict, idx: int) -> Optional[QAExample]:
        """Convert a raw PsiloQA row into a :class:`QAExample`."""
        qid = f"psiloqa:{idx}"
        knowledge = str(row.get("wiki_passage", "") or "")
        question = str(row.get("question", "") or "")
        right_answer = str(row.get("golden_answer", "") or "")

        if not question:
            return None

        return QAExample(
            qid=qid,
            knowledge=knowledge,
            question=question,
            right_answer=right_answer,
        )

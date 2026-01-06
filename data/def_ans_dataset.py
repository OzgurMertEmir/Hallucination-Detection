"""
def_ans_dataset
===============

Wrapper for the DefAn hallucination evaluation dataset on Hugging Face.

The dataset provides question/answer pairs; we map each row to a
:class:`~hallucination_detector.data.examples.QAExample` using the question
as the prompt and the provided answer as the ground truth. No hallucinated
answer is supplied by the dataset.

Example
-------

>>> from hallucination_detector.data.def_ans_dataset import DefAnDataset
>>> ds = DefAnDataset(split="train")
>>> ex = next(ds.examples())
>>> ex.right_answer
'Emil von Behring'
"""

from __future__ import annotations

import logging
import json
from urllib.request import urlopen
from typing import Dict, Iterable, Optional, Sequence

from data.examples import QAExample

try:
    from datasets import Dataset  # type: ignore
except Exception:
    Dataset = None  # type: ignore

logger = logging.getLogger(__name__)


class DefAnDataset:
    """Load and iterate over the DefAn dataset.

    Parameters
    ----------
    split : str, optional
        Dataset split to load. Defaults to ``"train"``. The upstream
        dataset exposes a single split.
    dataset_name : str, optional
        Identifier passed to :func:`datasets.load_dataset`. Override to
        point to a local copy if needed.
    """

    DEFAULT_DATASET = "iamasQ/DefAn"
    BASE_FILE_URL = "https://huggingface.co/datasets/iamasQ/DefAn/resolve/main/QA_domain_{idx}_public.json"

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = DEFAULT_DATASET,
        domain_indices: Sequence[int] = tuple(range(1, 9)),
    ) -> None:
        # The upstream repo does not ship a dataset script, so we load the
        # published JSON files directly.
        self.split = split.lower()
        self.dataset_name = dataset_name
        self.domain_indices = domain_indices
        self.data: Optional[Iterable[Dict]] = None
        self._length: int = 0
        self._load()

    def _load(self) -> None:
        if Dataset is None:
            raise ImportError(
                "datasets library is not available. Please install datasets to use DefAnDataset."
            )
        if self.split not in ("train", "default", "all"):
            raise ValueError(f"Unsupported split '{self.split}' for DefAnDataset; only 'train' is available.")

        data_files = [self.BASE_FILE_URL.format(idx=i) for i in self.domain_indices]
        rows = []
        try:
            for url in data_files:
                with urlopen(url) as resp:
                    payload = json.load(resp)
                    if isinstance(payload, list):
                        rows.extend(payload)
                    elif isinstance(payload, dict):
                        # Sometimes files wrap records in a top-level key.
                        # Attempt to flatten common patterns.
                        for value in payload.values():
                            if isinstance(value, list):
                                rows.extend(value)
                                break
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DefAn dataset from published JSON files. Ensure you have an internet "
                "connection and that the `datasets` package is installed."
            ) from exc
        if not rows:
            raise RuntimeError("Loaded zero records from DefAn; verify the source URLs.")
        # Normalise raw rows into a simple list to avoid Arrow type issues.
        cleaned_rows = []
        for row in rows:
            question = str(row.get("question") or row.get("questions") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not question or not answer:
                continue
            cleaned_rows.append({"question": question, "answer": answer})
        self.data = cleaned_rows
        self._length = len(self.data)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def examples(self, max_examples: Optional[int] = None) -> Iterable[QAExample]:
        """Yield dataset rows as :class:`QAExample` objects."""
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
        """Convert a raw DefAn row into a :class:`QAExample`."""
        qid = f"defan:{idx}"
        question = str(row.get("question") or row.get("questions") or "").strip()
        right_answer = str(row.get("answer") or "").strip()

        if not question or not right_answer:
            return None

        return QAExample(
            qid=qid,
            knowledge="",
            question=question,
            right_answer=right_answer,
        )

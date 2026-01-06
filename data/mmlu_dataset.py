"""
mmlu_dataset
============

Wrapper for the MMLU-Pro dataset hosted on the Hugging Face Hub.

The `TIGER-Lab/MMLU-Pro` dataset comprises multiple–choice questions
across a wide range of subjects.  Each example contains a question,
a list of answer options, an answer identifier, and a chain–of–thought
explanation.  This module exposes each record as a
 :class:`~hallucination_detector.data.examples.QAExample` for use in
 downstream processing.

The MMLU dataset stores both an ``answer`` letter (e.g. "A") and an
``answer_index`` integer.  In the dataset, ``answer_index`` is
1‑based (i.e. 1 corresponds to the first option).  This wrapper
converts whichever field is provided into a zero‑based index.  If
neither can be interpreted, the record is skipped.

Example
-------

>>> from hallucination_detector.data.mmlu_dataset import MMLUDataset
>>> ds = MMLUDataset(split="validation")
>>> ex = next(ds.examples())
>>> ex.question
'Which Treaty ended World War I?'
>>> ex.right_answer
'the Treaty of Versailles'

"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Dict, List

from datasets import load_dataset

from data.examples import QAExample

logger = logging.getLogger(__name__)


class MMLUDataset:
    """Load and iterate over the MMLU-Pro multiple–choice dataset.

    Parameters
    ----------
    split : str, optional
        The dataset split to load.  Valid values typically include
        ``"train"``, ``"validation"``, ``"test"`` and ``"dev"``.  Defaults
        to ``"validation"``.

    Notes
    -----
    Upon instantiation, this class downloads the dataset via
    :func:`datasets.load_dataset`.  If the download fails due to
    network issues or missing dependencies, a :class:`RuntimeError` is
    raised.
    """

    def __init__(self, split: str = "test") -> None:
        self.split = split
        try:
            self.data = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MMLU-Pro split '{split}'. Ensure you have an internet connection "
                f"and that the `datasets` package is installed."
            ) from exc

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def examples(self, max_examples: Optional[int] = None) -> Iterable[QAExample]:
        """Yield examples from the dataset as :class:`QAExample` objects.

        Parameters
        ----------
        max_examples : int, optional
            Optionally limit the number of examples produced.  Useful
            when sampling large datasets.

        Yields
        ------
        QAExample
            A normalised question/answer pair with optional
            knowledge and a hallucinated answer chosen from the
            incorrect options.
        """
        count = 0
        for i, row in enumerate(self.data):
            ex = self._normalize(row, i)
            if ex is None:
                continue
            yield ex
            count += 1
            if max_examples is not None and count >= max_examples:
                break

    def _parse_answer_index(self, row: Dict) -> Optional[int]:
        """Determine the zero‑based index of the correct answer.

        The dataset may supply either an ``answer_index`` field or an
        ``answer`` letter.  ``answer_index`` is 1‑based, so we
        subtract one to obtain a zero‑based index.  If an answer
        cannot be parsed, return ``None``.
        """
        # Prefer explicit index if available
        if "answer_index" in row and row["answer_index"] is not None:
            try:
                idx = int(row["answer_index"])
            except Exception:
                idx = None
            if idx is not None and 1 <= idx <= 100:  # guard against invalid values
                return idx - 1
        # Fallback to letter
        answer_letter = (row.get("answer") or "").strip().upper()
        if answer_letter:
            # Map letters A–Z to zero‑based indices
            # Only map up to the number of provided options
            letter_ord = ord(answer_letter) - ord("A")
            if 0 <= letter_ord:
                return letter_ord
        return None

    def build_options_text(self, options):
        option_texts = []
        for i, option in enumerate(options):
            label = chr(ord('A')+i)
            option_text = f"{label}) {option}"
            option_texts.append(option_text)

        return "\n".join(option_texts)


    def _normalize(self, row: Dict, i) -> Optional[QAExample]:
        """Convert a raw dataset row into a :class:`QAExample`.

        Returns ``None`` if the record cannot be parsed.
        """
        options: List[str] = list(row.get("options") or [])
        options_text = self.build_options_text(options)
        right_answer = (row.get("answer") or "").strip()
        knowledge = ""
        question = (row.get("question") or "").strip()
        return QAExample(
            qid=f"mmlu:{i}",
            knowledge=knowledge,
            question=question,
            right_answer=right_answer,
            options_text=options_text
        )

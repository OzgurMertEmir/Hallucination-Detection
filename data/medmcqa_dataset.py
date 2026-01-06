"""
medmcqa_dataset
================

Wrapper for the MedMCQA dataset hosted on the Hugging Face Hub.

This module exposes questions from the
``openlifescienceai/medmcqa`` dataset as :class:`~hallucination_detector.data.examples.QAExample`
objects.  Each dataset entry represents a single multiple–choice
medical question with four possible answers.  A ``cop`` field
indicates the correct option, and an optional ``exp`` field provides
an explanation which we treat as additional knowledge.

Only single–choice questions are supported; rows marked as
``multi`` are skipped.  The correct answer is extracted from the
``cop`` field, which may be stored as an integer (1–4) or as a
string containing other tokens (such as difficulty).  Any row where
the correct answer cannot be parsed is skipped.

Example
-------

>>> from hallucination_detector.data.medmcqa_dataset import MedMCQADataset
>>> ds = MedMCQADataset(split="validation")
>>> ex = next(ds.examples())
>>> ex.question
'Which gas is used as a coolant in nuclear reactors?'
>>> ex.right_answer
'Carbon dioxide'

"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Dict, List

from datasets import load_dataset

from .examples import QAExample

logger = logging.getLogger(__name__)


class MedMCQADataset:
    """Load and iterate over the MedMCQA dataset.

    Parameters
    ----------
    split : str, optional
        The dataset split to load.  Valid values include
        ``"train"``, ``"validation"`` and ``"test"``.  Defaults to
        ``"train"``.

    Notes
    -----
    When instantiated, the dataset is downloaded from the Hugging Face
    hub via :func:`datasets.load_dataset`.  If the download fails
    (for example, due to lack of internet connectivity), a
    :class:`RuntimeError` is raised.
    """

    def __init__(self, split: str = "train") -> None:
        self.split = split
        try:
            self.data = load_dataset("openlifescienceai/medmcqa", split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MedMCQA split '{split}'. Ensure you have an internet connection "
                f"and that the `datasets` package is installed."
            ) from exc

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def examples(self, max_examples: Optional[int] = None) -> Iterable[QAExample]:
        """Yield normalised question–answer pairs as :class:`QAExample` objects.

        Parameters
        ----------
        max_examples : int, optional
            If provided, limit the number of examples yielded.  Useful
            for debugging or sampling.

        Yields
        ------
        QAExample
            A single question/answer pair with optional knowledge and
            a hallucinated answer chosen from the incorrect options.
        """
        count = 0
        for row in self.data:
            ex = self._normalize(row)
            if ex is None:
                continue
            yield ex
            count += 1
            if max_examples is not None and count >= max_examples:
                break

    def _parse_correct_index(self, cop_value: object) -> Optional[int]:
        """Parse the ``cop`` field and return a zero–based index of the correct option.

        The ``cop`` field can be either an integer (1–4) or a string
        containing arbitrary tokens followed by a letter or number.  This
        helper extracts the last alphanumeric token and maps it to 0–3.

        Returns
        -------
        int or None
            The zero–based index of the correct answer, or ``None`` if
            parsing fails.
        """
        # Integer format: 1‑based indexing
        if isinstance(cop_value, int):
            if 1 <= cop_value <= 4:
                return cop_value - 1
            return None
        # Convert to string and process
        cop_str = str(cop_value).strip().lower()
        if not cop_str:
            return None
        tokens = cop_str.split()
        last = tokens[-1]
        # Numeric token
        if last.isdigit():
            idx = int(last)
            if 1 <= idx <= 4:
                return idx - 1
        # Alphabetic token
        if last.isalpha():
            mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
            return mapping.get(last)
        return None

    def _normalize(self, row: Dict) -> Optional[QAExample]:
        """Convert a raw dataset record into a :class:`QAExample`.

        Rows that specify multiple correct answers (``choice_type`` equal
        to ``"multi"``) are skipped.  If the correct answer cannot be
        determined, ``None`` is returned.
        """
        # Skip multi–answer questions
        if row.get("choice_type", "single").lower() != "single":
            return None
        idx = self._parse_correct_index(row.get("cop"))
        if idx is None:
            return None
        # Retrieve options; some entries may be None
        options: List[str] = [
            row.get("opa") or "",
            row.get("opb") or "",
            row.get("opc") or "",
            row.get("opd") or "",
        ]
        if not (0 <= idx < len(options)):
            return None
        right_answer = options[idx].strip()
        # Choose a hallucinated answer by picking the first incorrect option
        wrong_idx = 0 if idx != 0 else 1
        hallucinated_answer = options[wrong_idx].strip() if options[wrong_idx] else ""
        knowledge = (row.get("exp") or "").strip()
        question = (row.get("question") or "").strip()
        return QAExample(
            knowledge=knowledge,
            question=question,
            right_answer=right_answer,
            hallucinated_answer=hallucinated_answer,
        )
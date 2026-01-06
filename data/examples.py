from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class QAExample:
    """A single question–answer pair with ground truth information."""
    qid: str
    knowledge: str
    question: str
    right_answer: str
    hallucinated_answer: Optional[str] = None
    options_text: Optional[str] = None
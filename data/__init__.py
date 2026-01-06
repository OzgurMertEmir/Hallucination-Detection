"""Expose dataset wrappers and utility classes for training and evaluation."""

from .examples import QAExample
from .def_ans_dataset import DefAnDataset
from .halueval_dataset import HaluEvalDataset
from .medmcqa_dataset import MedMCQADataset
from .mmlu_dataset import MMLUDataset
from .psiloqa_dataset import PsiloQADataset

__all__ = [
    "QAExample",
    "DefAnDataset",
    "HaluEvalDataset",
    "MedMCQADataset",
    "MMLUDataset",
    "PsiloQADataset",
    "DATASET_REGISTRY",
]


DATASET_REGISTRY = {
    "DefAnDataset": DefAnDataset,
    "HaluEvalDataset": HaluEvalDataset,
    "MedMCQADataset": MedMCQADataset,
    "MMLUDataset": MMLUDataset,
    "PsiloQADataset": PsiloQADataset,
}

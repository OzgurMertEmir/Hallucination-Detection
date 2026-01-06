"""
JudgeLabeler: lightweight LLM-as-a-Judge labeling using UQLM.UQEnsemble (judges only).

- You pass LangChain chat models (e.g., AzureChatOpenAI, ChatOpenAI, ChatVertexAI) as judges.
- Each judge scores a candidate answer on a 0/0.5/1 scale (incorrect/uncertain/correct) using UQLM's templates.
- We average the judge scores (weighted if you provide weights) to get `ensemble_score`.
- We convert to a binary label:
      label = 1  (hallucinated)   if ensemble_score < thresh_low
      label = 0  (not hallucinated) if ensemble_score >= thresh_high
      abstain if thresh_low <= score < thresh_high (optional)

Smart prompt:
- We pack Question + (optional) Reference Answer into the "prompt" so judges can compare against GT.
- Candidate (model) answer is fed as the "response".

Dependencies:
    pip install uqlm langchain

Typical usage:
    from langchain_openai import AzureChatOpenAI
    judges = [AzureChatOpenAI(...), ...]
    labeler = JudgeLabeler(judges=judges, config=JudgeLabelerConfig())
    y, info = labeler.label(question, model_answer, reference_answer)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from .prompts import prompt_for_labeler_llm, labeler_sys_prompt
import logging

try:
    from uqlm import LLMPanel
    import asyncio
except Exception as e:
    LLMPanel = None

logger = logging.getLogger(__name__)

class JudgeLabeler:
    """
    Minimal, effective judge-only labeler.
    """

    def __init__(
        self,
        judges: List[Any],
    ):
        self.panel = LLMPanel(judges=judges, explanations=True)
  
    async def label(
        self,
        ex,
        model_answer: str,
        *,
        show_progress_bars: bool = False,
    ) -> Tuple[int, Dict[str, float]]:
        prompt = prompt_for_labeler_llm(ex)
        logger.info("Prompt: %s\nModel Response: %s\n", prompt, model_answer)
        res = await self.panel.score(prompts=[prompt], responses=[model_answer], show_progress_bars=show_progress_bars)
        df = res.to_df()

        score = float(df.loc[0, "judge_1"])
        explanation = df.loc[0, "judge_1_explanation"]
        logger.info("Correctness Score of the Judge: %s\nExplanation of the models judgment: %s", score, explanation)
        label = (score + 1) % 2
        return label

    async def batch_labels(
        self,
        prompts,
        model_answers,
        *,
        show_progress_bars: bool = False
    ):
        res = await self.panel.score(prompts=prompts, responses=model_answers, show_progress_bars=show_progress_bars)
        df = res.to_df()
        labels = [(int(row['judge_1']) + 1) % 2 for index, row in df.iterrows()]
        explanations = [row['judge_1_explanation'] for index, row in df.iterrows()]
        return labels
        

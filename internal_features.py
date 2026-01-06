
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
from contextlib import nullcontext

try:
    import torch, asyncio
    from langchain_openai import ChatOpenAI
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    ChatOpenAI = None

from data import QAExample
from prompts import prompt_for_model_answer, prompt_for_labeler_llm
from labeling_judge import JudgeLabeler

logger = logging.getLogger(__name__)


@dataclass
class InternalExampleState:
    """Container for internal states of one (prompt, answer)."""
    qid: str
    question: str
    answer: str
    label: str

    question_ids: List[int]
    answer_ids: List[int]

    hidden_states: List["torch.Tensor"]   # list of (T, d)
    attentions: List["torch.Tensor"]      # list of (H, T, T)
    ffn_hidden: "torch.Tensor"            # (T, d)
    logits: "torch.Tensor"                # (T, V)

    seq_len: int
    q_len: int
    a_len: int


@dataclass
class PreparedBatch:
    """Tokenized inputs ready for generation."""
    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"


class InternalStateExtractor:
    """Generate answers (batched) and capture internal states in a single pass."""

    def __init__(
        self,
        model_name_or_path: str,
        judge_model_name: Optional[str] = "gpt-4o-mini",
        device: str = "cpu",
        batch_size: int = 4,
        max_new_tokens: int = 64,
        capture_dtype: Optional[str] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_new_tokens = max_new_tokens
        if capture_dtype is None:
            if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
                capture_dtype = "float16"
            else:
                capture_dtype = "float32"
        self.capture_dtype = capture_dtype

        self.model = None
        self.tokenizer = None
        self._load_model()

        self.judge = None
        self.load_judge(judge_model_name)

    # === Model loading =======================================================

    def _load_model(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            logger.warning("transformers/torch not available; InternalStateExtractor will be inert.")
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, attn_implementation="eager").to(self.device).eval()


    def load_judge(self, openai_model_name):
        if ChatOpenAI is None:
            logger.warning(
                "ChatOpenAI is not available.  Falling back to no judge labeler; hallucination labels will use dataset annotations."
            )
        else:
            judges = [
                ChatOpenAI(
                    model=openai_model_name,
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                ),
            ]
            self.judge = JudgeLabeler(judges=judges)

    def _autocast_context(self):
        if torch is None:
            return nullcontext()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def prepare(self, batch: List[QAExample]) -> PreparedBatch:
        """
        Tokenize the batch prompts on CPU so that generation can start immediately
        once the GPU is available.
        """
        if torch is None or self.tokenizer is None:
            raise RuntimeError("Torch/tokenizer not available; cannot prepare batch.")
        prompts = [prompt_for_model_answer(ex) for ex in batch]
        logger.info("Example Prompt:\n{\n%s\n}", prompts[0])
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        return PreparedBatch(
            input_ids=enc["input_ids"].clone(),
            attention_mask=enc["attention_mask"].clone(),
        )

    async def get_labels(self, batch: List[QAExample], answers_text: List[str]) -> List[int]:
        if self.judge is not None:
            judge_prompts = [prompt_for_labeler_llm(ex) for ex in batch]
            labels = await self.judge.batch_labels(judge_prompts, answers_text)
            return labels
        else:
            logger.info("No judge model")
            return

    # === Public API ==========================================================

    def extract(self, batch: List[QAExample], prepared: Optional[PreparedBatch] = None) -> List[InternalExampleState]:
        if torch is None or self.model is None or self.tokenizer is None:
            # Minimal fallback: return empty states with dataset answers
            out: List[InternalExampleState] = []
            for ex in batch:
                ans = ex.hallucinated_answer or ex.right_answer
                out.append(
                    InternalExampleState(
                        question=ex.question,
                        answer=ans,
                        question_ids=[],
                        answer_ids=[],
                        hidden_states=[],
                        attentions=[],
                        ffn_hidden=torch.zeros(0),
                        logits=torch.zeros(0),
                        seq_len=0, q_len=0, a_len=0,
                    )
                )
            return out

        # 1) Build prompts and tokenize (left pad for generation)
        if prepared is None:
            prepared = self.prepare(batch)
        input_ids = prepared.input_ids.to(self.device)
        attn_mask = prepared.attention_mask.to(self.device)

        # 2) Batched greedy generation
        with torch.inference_mode():
            # AMP autocast if CUDA is used
            with self._autocast_context():
                gen = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                )

        # Decode answers and collect prompt/answer ids per example
        answers_text: List[str] = []
        q_ids_list: List[List[int]] = []
        a_ids_list: List[List[int]] = []

        for i in range(len(batch)):
            # prompt length for this row
            row_len    = int(input_ids.size(1))            # includes left PADs
            prompt_len = int(attn_mask[i].sum().item())    # non-PAD prompt tokens
            out_ids    = gen[i]
            a_ids      = out_ids[row_len:].tolist()        # new tokens
            q_ids      = out_ids[row_len - prompt_len : row_len].tolist()  # prompt w/o PADs
            text       = self.tokenizer.decode(a_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            answers_text.append(text)
            q_ids_list.append(q_ids)
            a_ids_list.append(a_ids)

        if self.judge is not None:
            judge_prompts = [prompt_for_labeler_llm(ex) for ex in batch]
            logger.info("Judge Prompt Example:\n{\n%s\n}\nModels Answer:\n{\n%s\n}", judge_prompts[0], answers_text[0])
            labels =  asyncio.run(self.judge.batch_labels(judge_prompts, answers_text))
        else:
            # Fallback: use dataset-provided labels when an external judge is unavailable.
            labels = []
            for ex, answer_text in zip(batch, answers_text):
                if ex.hallucinated_answer is not None:
                    labels.append(int(answer_text.strip() != ex.right_answer.strip()))
                else:
                    labels.append(0)

        # 3) Build full sequences (prompt + generated answer) for internal pass
        full_ids = [torch.tensor(q + a, dtype=torch.long) for q, a in zip(q_ids_list, a_ids_list)]
        max_len = max(t.numel() for t in full_ids) if full_ids else 0
        pad_id = int(self.tokenizer.pad_token_id or 0)
        ids_batch = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.numel()), value=pad_id) for t in full_ids]).to(self.device)
        mask_batch = (ids_batch != pad_id).long().to(self.device)

        # 4) Single forward pass to capture hidden/attn/logits/FFN via hook
        with FFNHookGrabber(self.model) as grabber:
            with torch.inference_mode():
                #with self._autocast_context():
                outputs = self.model(
                    input_ids=ids_batch,
                    attention_mask=mask_batch,
                    output_hidden_states=True,
                    output_attentions=True,
                )
        hidden_all = outputs.hidden_states  # tuple L+1 of (B, T, d)
        attn_all = outputs.attentions      # tuple L of (B, H, T, T)
        logits_all = outputs.logits        # (B, T, V)
        ffn_all = grabber.tensor           # (B, T, d)

        # 5) Labels (simple lexical fallback if judge is not wired)
        states: List[InternalExampleState] = []
        for i, ex in enumerate(batch):
            q_ids = q_ids_list[i]
            a_ids = a_ids_list[i]
            T = len(q_ids) + len(a_ids)
            # slice per-sample tensors
            h_list = [H[i, :T].detach() for H in hidden_all]
            a_list = [A[i, :, :T, :T].detach() for A in attn_all]
            logits = logits_all[i, :T].detach()
            ffn = ffn_all[i, :T].detach() if isinstance(ffn_all, torch.Tensor) else torch.zeros((T, h_list[-1].size(-1)), device=self.device)

            ans_text = answers_text[i]
            label = labels[i]

            #logger.info("Prompt:\n{\n%s\n}\nModels Output:\n{\n%s\n}\nJudges Label:{%s}", judge_prompts[i], ans_text, label)
            target_dtype = getattr(torch, self.capture_dtype, torch.float16 if self.device.startswith("cuda") else torch.float32)
            states.append(
                InternalExampleState(
                    qid=ex.qid,
                    question=ex.question,
                    answer=ans_text,
                    label=label,
                    question_ids=q_ids,
                    answer_ids=a_ids,
                    hidden_states=[t.to(dtype=target_dtype) for t in h_list],
                    attentions=[t.to(dtype=target_dtype) for t in a_list],
                    ffn_hidden=ffn.to(dtype=target_dtype),
                    logits=logits.to(dtype=target_dtype),
                    seq_len=T, q_len=len(q_ids), a_len=len(a_ids),
                )
            )
        return states


class FFNHookGrabber:
    """
    Register a hook on a representative FFN/MLP module to capture activations.
    For generality, we try multiple known attribute paths.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.handle = None
        self.tensor = None
        # candidates inside a transformer block
        self.ffn_attr_candidates = [
            "mlp",
            "ffn",
            "feed_forward",
            "ff",
            "MlpBlock",
            "dense_ff",
        ]

    def _resolve_blocks(self):
        m = self.model
        # common model families
        for path in [
            "model.layers",           # LLaMA/OPT
            "transformer.h",          # GPT-2/Neo
            "gpt_neox.layers",        # NeoX
            "model.decoder.layers",   # OPT/BART decoder
            "layers",                 # fallback
        ]:
            try:
                obj = m
                for p in path.split("."):
                    obj = getattr(obj, p)
                return obj
            except Exception:
                continue
        raise AttributeError("Could not find transformer blocks on model")

    def _resolve_ffn_module(self):
        blocks = self._resolve_blocks()
        block = blocks[len(blocks)//2]  # middle block as heuristic
        for cand in self.ffn_attr_candidates:
            if hasattr(block, cand):
                sub = getattr(block, cand)
                # find the last linear in the FFN
                for name in ["out_proj", "c_proj", "down_proj", "fc2"]:
                    if hasattr(sub, name):
                        return getattr(sub, name)
                return sub
        # fallback: try to find first Linear child
        for n, mod in block.named_modules():
            if mod.__class__.__name__.lower().find("linear") >= 0:
                return mod
        raise AttributeError("Could not resolve FFN module")

    def __enter__(self):
        try:
            ffn = self._resolve_ffn_module()
        except Exception as e:
            logger.warning("FFN resolution failed: %s; using identity hook on model output", e)
            def _hook(_m, _inp, out):
                self.tensor = out.detach()
            self.handle = self.model.register_forward_hook(_hook)
            return self

        def hook(_module, _inp, out):
            # expect out shape (B, T, d)
            self.tensor = out.detach()
        self.handle = ffn.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

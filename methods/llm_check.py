# methods/llm_check_new.py
"""
LLM-Check (NeurIPS'24) feature extraction, wired directly to the official repo utilities.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import logging

try:
    import torch
except Exception:
    torch = None

# Project types
from .method_cfg import MethodConfig
from ..internal_features import InternalExampleState
from .LLM_Check_Hallucination_Detection import common_utils as cu

logger = logging.getLogger(__name__)


# ----------------------------
# Minimal, paper-aligned config
# ----------------------------
@dataclass
class LLMCheckConfig(MethodConfig):
    """
    Parameters (mirror common_utils defaults / paper choices):
    - entropy_top_k:    top-k for logit entropy (repo often uses 50)
    - window_size:      window size 'w' for windowed logit entropy (repo uses w=1 in compute_scores)
    - answer_only:      compute logits-based scores only on the answer part of the sequence
    - hidden_layer_idx: if set, compute Hidden score for exactly this layer;
                        if None, compute per-layer scores for all layers [1..L-1] and return mean/std + the list
    - attn_layer_idx:   same, but for Attention score
    """
    entropy_top_k: int = 50
    window_size: int = 1
    answer_only: bool = True
    hidden_layer_idx: Optional[int] = None
    attn_layer_idx: Optional[int] = None

    def __init__(
        self,
        entropy_top_k: int = 50,
        window_size: int = 1,
        answer_only: bool = True,
        hidden_layer_idx: Optional[int] = None,
        attn_layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.entropy_top_k = int(entropy_top_k)
        self.window_size = int(window_size)
        self.answer_only = bool(answer_only)
        self.hidden_layer_idx = hidden_layer_idx if hidden_layer_idx is None else int(hidden_layer_idx)
        self.attn_layer_idx = attn_layer_idx if attn_layer_idx is None else int(attn_layer_idx)


# ----------------------------
# Public API
# ----------------------------
class LLMCheckFeatures:
    """
    Thin adapter that:
      1) aligns the token range (answer region by default),
      2) calls repo helpers (perplexity, logit entropy, windowed entropy, hidden SVD score, attention score),
      3) returns a flat dict with stable keys for your pipeline.
    """

    def compute(self, st: InternalExampleState, cfg: LLMCheckConfig) -> Dict[str, Any]:
        if torch is None:
            return {"label": int(st.label)}

        out: Dict[str, Any] = {"qid": st.qid, "label": int(st.label)}

        # =========
        # Prepare inputs expected by common_utils
        # =========
        device = None
        if torch is not None:
            if st.logits is not None and isinstance(st.logits, torch.Tensor):
                device = st.logits.device
            elif st.hidden_states and isinstance(st.hidden_states, list) and len(st.hidden_states) > 0:
                device = st.hidden_states[0].device
        if device is None:
            device = torch.device("cpu") if torch is not None else "cpu"

        # Token ids as a (1, T) tensor. We reconstruct from question+answer ids.
        tok_ids: List[int] = (st.question_ids or []) + (st.answer_ids or [])
        T = int(st.seq_len) if hasattr(st, "seq_len") else len(tok_ids)
        if not tok_ids and T > 0:
            # Fallback: if ids aren't stored, build a dummy tensor to avoid shape errors in repo funcs.
            tok_ids = [0] * T
        tok_in = torch.tensor([tok_ids], dtype=torch.long, device=device)

        # Token range (start, end) for scoring:
        # The repo indices expect that logits[t] predicts token t+1, and perplexity uses indices [i1:i2]
        # with logits positions [i1-1 : i2-1]. We'll pass the answer span by default.
        q_len = int(getattr(st, "q_len", 0) or 0)
        a_len = int(getattr(st, "a_len", 0) or 0)
        end_idx = q_len + a_len if a_len > 0 else T
        start_idx = q_len if cfg.answer_only else 0

        # To avoid negative indexing at the first element (i1-1), clamp start at 1 if necessary.
        # If your data always has q_len >= 1 this is a no-op; it keeps parity with repo formula.
        start_idx = max(start_idx, 1) if end_idx > 0 else 0

        tok_lens: List[Tuple[int, int]] = [(start_idx, end_idx)]
        tok_ins = [tok_in]  # list of (1, T) tensors, one per sample

        # Logits list: each element is (T, V) for a sample
        logits_list: List[torch.Tensor] = []
        if st.logits is not None and hasattr(st.logits, "dim") and st.logits.dim() == 2:
            logits_list = [st.logits.to(torch.float32)]
        else:
            logger.debug("LLM-Check: logits missing; skipping logits-based scores.")

        # Hidden states list: repo expects a list of tuples, each tuple is hidden states across layers
        hidden_acts: List[Tuple[torch.Tensor, ...]] = []
        if st.hidden_states and isinstance(st.hidden_states, list) and len(st.hidden_states) > 0:
            # Use as-is; many models have embeddings at index 0; repo functions handle layer indexing explicitly.
            hidden_acts = [tuple(h.to(torch.float32) for h in st.hidden_states)]
        else:
            logger.debug("LLM-Check: hidden states missing; skipping hidden-based scores.")

        # Attention list: repo expects a list of tuples, each tuple is attentions across layers
        attns: List[Tuple[torch.Tensor, ...]] = []
        if st.attentions and isinstance(st.attentions, list) and len(st.attentions) > 0:
            attns = [tuple(a.to(torch.float32) for a in st.attentions)]
        else:
            logger.debug("LLM-Check: attentions missing; skipping attention-based scores.")

        # =========
        # Logits-derived scores (perplexity, entropy, windowed entropy)
        # =========
        if logits_list:
            try:
                ppl_arr = cu.perplexity(logits_list, tok_ins, tok_lens, min_k=None)
                out["llmcheck_ppl"] = float(ppl_arr[0])
            except Exception as e:
                logger.warning("LLM-Check: perplexity() failed: %s", e)

            try:
                ent_arr = cu.logit_entropy(logits_list, tok_lens, top_k=cfg.entropy_top_k)
                out["llmcheck_logit_entropy"] = float(ent_arr[0])
            except Exception as e:
                logger.warning("LLM-Check: logit_entropy() failed: %s", e)

            try:
                # In the repo's compute_scores they call window_logit_entropy(..., w=1) (no top_k).
                win_arr = cu.window_logit_entropy(logits_list, tok_lens, top_k=None, w=cfg.window_size)
                out["llmcheck_window_entropy"] = float(win_arr[0])
            except Exception as e:
                logger.warning("LLM-Check: window_logit_entropy() failed: %s", e)

        # =========
        # Hidden-state SVD score
        # =========
        if hidden_acts:
            try:
                if cfg.hidden_layer_idx is not None:
                    hid_arr = cu.get_svd_eval(hidden_acts, layer_num=int(cfg.hidden_layer_idx),
                                              tok_lens=tok_lens, use_toklens=True)
                    out["llmcheck_hidden_score"] = float(hid_arr[0])
                else:
                    # Compute per-layer scores for layers [1..L-1] (skip embeddings at 0)
                    L_layers = len(hidden_acts[0])
                    hid_scores: List[float] = []
                    for layer_num in range(1, L_layers):
                        s = cu.get_svd_eval(hidden_acts, layer_num=layer_num,
                                            tok_lens=tok_lens, use_toklens=True)[0]
                        hid_scores.append(float(s))
                    if hid_scores:
                        out["llmcheck_hidden_scores"] = hid_scores
                        out["llmcheck_hidden_mean"] = float(torch.tensor(hid_scores).mean().item())
                        out["llmcheck_hidden_std"] = float(torch.tensor(hid_scores).std(unbiased=False).item()
                                                          if len(hid_scores) > 1 else 0.0)
            except Exception as e:
                logger.warning("LLM-Check: get_svd_eval() failed: %s", e)

        # =========
        # Attention eigenvalue score
        # =========
        if attns:
            try:
                # number of heads (H)
                H = int(st.attentions[0].shape[0]) if (st.attentions and hasattr(st.attentions[0], "shape")) else None

                if cfg.attn_layer_idx is not None:
                    att_arr = cu.get_attn_eig_prod(
                        [tuple(st.attentions)], layer_num=int(cfg.attn_layer_idx),
                        tok_lens=tok_lens, use_toklens=True
                    )
                    s = float(att_arr[0])                  # repo’s “sum over heads of mean log(diag)”
                    out["llmcheck_attn_score"] = s

                    if H and H > 0:
                        # 1) per-head mean (still in log space; more interpretable across models with diff head counts)
                        s_headmean = s / H
                        out["llmcheck_attn_headmean"] = float(s_headmean)

                        # 2) geometric self-attention mass in (0,1]: exp(mean log diag per head)
                        s_geom = float(torch.exp(torch.tensor(s_headmean)))
                        out["llmcheck_attn_geom"] = s_geom

                else:
                    L_layers = len(attns[0])
                    att_scores: List[float] = []
                    for layer_num in range(1, L_layers):
                        s = cu.get_attn_eig_prod(
                            [tuple(st.attentions)], layer_num=layer_num,
                            tok_lens=tok_lens, use_toklens=True
                        )[0]
                        att_scores.append(float(s))

                    if att_scores:
                        # original repo-faithful outputs
                        out["llmcheck_attn_scores"] = att_scores
                        att_t = torch.tensor(att_scores, dtype=torch.float32)
                        out["llmcheck_attn_mean"] = float(att_t.mean().item())
                        out["llmcheck_attn_std"]  = float(att_t.std(unbiased=False).item()) if att_t.numel() > 1 else 0.0

                        if H and H > 0:
                            # 1) per-head mean (per layer)
                            headmean_t = att_t / float(H)                    # (L-1,)
                            out["llmcheck_attn_headmean_scores"] = headmean_t.tolist()
                            out["llmcheck_attn_headmean_mean"]   = float(headmean_t.mean().item())
                            out["llmcheck_attn_headmean_std"]    = float(headmean_t.std(unbiased=False).item()) if headmean_t.numel() > 1 else 0.0

                            # 2) geometric self-attention mass (per layer) in (0,1]
                            geom_t = torch.exp(headmean_t)                   # exp(mean log) = geometric mean
                            out["llmcheck_attn_geom_scores"] = geom_t.tolist()
                            out["llmcheck_attn_geom_mean"]   = float(geom_t.mean().item())
                            out["llmcheck_attn_geom_std"]    = float(geom_t.std(unbiased=False).item()) if geom_t.numel() > 1 else 0.0

            except Exception as e:
                logger.warning("LLM-Check: get_attn_eig_prod() failed: %s", e)

        return out

    def compute_batch(self, states: List[InternalExampleState], cfg: LLMCheckConfig) -> List[Dict[str, Any]]:
        if torch is None:
            return [{"label": int(st.label)} for st in states]

        rows: List[Dict[str, Any]] = [{"qid": st.qid, "label": int(st.label)} for st in states]

        tok_ins_all: List[torch.Tensor] = []
        tok_lens_all: List[Tuple[int, int]] = []

        logits_entries: List[Tuple[int, torch.Tensor]] = []
        hidden_entries: List[Tuple[int, Tuple[torch.Tensor, ...]]] = []
        attn_entries: List[Tuple[int, Tuple[torch.Tensor, ...]]] = []

        for idx, st in enumerate(states):
            tok_ids: List[int] = (st.question_ids or []) + (st.answer_ids or [])
            T = int(st.seq_len) if hasattr(st, "seq_len") else len(tok_ids)
            if not tok_ids and T > 0:
                tok_ids = [0] * T

            device = None
            if st.logits is not None and isinstance(st.logits, torch.Tensor):
                device = st.logits.device
            elif st.hidden_states and isinstance(st.hidden_states, list) and st.hidden_states:
                device = st.hidden_states[0].device
            elif st.attentions and isinstance(st.attentions, list) and st.attentions:
                device = st.attentions[0].device
            else:
                device = torch.device("cpu")

            tok_in = torch.tensor([tok_ids], dtype=torch.long, device=device)
            q_len = int(getattr(st, "q_len", 0) or 0)
            a_len = int(getattr(st, "a_len", 0) or 0)
            end_idx = q_len + a_len if a_len > 0 else T
            start_idx = q_len if cfg.answer_only else 0
            start_idx = max(start_idx, 1) if end_idx > 0 else 0

            tok_ins_all.append(tok_in)
            tok_lens_all.append((start_idx, end_idx))

            if st.logits is not None and hasattr(st.logits, "dim") and st.logits.dim() == 2:
                logits_entries.append((idx, st.logits.to(torch.float32)))

            if st.hidden_states and isinstance(st.hidden_states, list) and st.hidden_states:
                hidden_entries.append((idx, tuple(h.to(torch.float32) for h in st.hidden_states)))

            if st.attentions and isinstance(st.attentions, list) and st.attentions:
                attn_entries.append((idx, tuple(a.to(torch.float32) for a in st.attentions)))

        if logits_entries:
            try:
                logits_list = [entry[1] for entry in logits_entries]
                tok_subset = [tok_ins_all[entry[0]] for entry in logits_entries]
                lens_subset = [tok_lens_all[entry[0]] for entry in logits_entries]

                ppl_arr = cu.perplexity(logits_list, tok_subset, lens_subset, min_k=None)
                ent_arr = cu.logit_entropy(logits_list, lens_subset, top_k=cfg.entropy_top_k)
                win_arr = cu.window_logit_entropy(logits_list, lens_subset, top_k=None, w=cfg.window_size)

                for idx_val, ppl_val, ent_val, win_val in zip(
                    [entry[0] for entry in logits_entries],
                    ppl_arr,
                    ent_arr,
                    win_arr,
                ):
                    rows[idx_val]["llmcheck_ppl"] = float(ppl_val)
                    rows[idx_val]["llmcheck_logit_entropy"] = float(ent_val)
                    rows[idx_val]["llmcheck_window_entropy"] = float(win_val)
            except Exception as exc:
                logger.warning("LLM-Check: batch logits metrics failed: %s", exc)

        if hidden_entries:
            for idx_val, hidden_tuple in hidden_entries:
                try:
                    if cfg.hidden_layer_idx is not None:
                        hid_arr = cu.get_svd_eval(
                            [hidden_tuple],
                            layer_num=int(cfg.hidden_layer_idx),
                            tok_lens=[tok_lens_all[idx_val]],
                            use_toklens=True,
                        )
                        rows[idx_val]["llmcheck_hidden_score"] = float(hid_arr[0])
                    else:
                        L_layers = len(hidden_tuple)
                        hid_scores: List[float] = []
                        for layer_num in range(1, L_layers):
                            s = cu.get_svd_eval(
                                [hidden_tuple],
                                layer_num=layer_num,
                                tok_lens=[tok_lens_all[idx_val]],
                                use_toklens=True,
                            )[0]
                            hid_scores.append(float(s))
                        if hid_scores:
                            rows[idx_val]["llmcheck_hidden_scores"] = hid_scores
                            rows[idx_val]["llmcheck_hidden_mean"] = float(torch.tensor(hid_scores).mean().item())
                            rows[idx_val]["llmcheck_hidden_std"] = float(
                                torch.tensor(hid_scores).std(unbiased=False).item()
                                if len(hid_scores) > 1 else 0.0
                            )
                except Exception as exc:
                    logger.warning("LLM-Check: hidden metrics failed for sample %d: %s", idx_val, exc)

        if attn_entries:
            for idx_val, attn_tuple in attn_entries:
                try:
                    H = int(attn_tuple[0].shape[0]) if attn_tuple and hasattr(attn_tuple[0], "shape") else None
                    if cfg.attn_layer_idx is not None:
                        att_arr = cu.get_attn_eig_prod(
                            [attn_tuple],
                            layer_num=int(cfg.attn_layer_idx),
                            tok_lens=[tok_lens_all[idx_val]],
                            use_toklens=True,
                        )
                        s = float(att_arr[0])
                        rows[idx_val]["llmcheck_attn_score"] = s
                        if H and H > 0:
                            s_headmean = s / H
                            rows[idx_val]["llmcheck_attn_headmean"] = float(s_headmean)
                            rows[idx_val]["llmcheck_attn_geom"] = float(torch.exp(torch.tensor(s_headmean)))
                    else:
                        L_layers = len(attn_tuple)
                        att_scores: List[float] = []
                        for layer_num in range(1, L_layers):
                            s = cu.get_attn_eig_prod(
                                [attn_tuple],
                                layer_num=layer_num,
                                tok_lens=[tok_lens_all[idx_val]],
                                use_toklens=True,
                            )[0]
                            att_scores.append(float(s))
                        if att_scores:
                            rows[idx_val]["llmcheck_attn_scores"] = att_scores
                            att_t = torch.tensor(att_scores, dtype=torch.float32)
                            rows[idx_val]["llmcheck_attn_mean"] = float(att_t.mean().item())
                            rows[idx_val]["llmcheck_attn_std"] = float(
                                att_t.std(unbiased=False).item()
                            ) if att_t.numel() > 1 else 0.0
                            if H and H > 0:
                                headmean_t = att_t / float(H)
                                rows[idx_val]["llmcheck_attn_headmean_scores"] = headmean_t.tolist()
                                rows[idx_val]["llmcheck_attn_headmean_mean"] = float(headmean_t.mean().item())
                                rows[idx_val]["llmcheck_attn_headmean_std"] = float(
                                    headmean_t.std(unbiased=False).item()
                                ) if headmean_t.numel() > 1 else 0.0
                                geom_t = torch.exp(headmean_t)
                                rows[idx_val]["llmcheck_attn_geom_scores"] = geom_t.tolist()
                                rows[idx_val]["llmcheck_attn_geom_mean"] = float(geom_t.mean().item())
                                rows[idx_val]["llmcheck_attn_geom_std"] = float(
                                    geom_t.std(unbiased=False).item()
                                ) if geom_t.numel() > 1 else 0.0
                except Exception as exc:
                    logger.warning("LLM-Check: attention metrics failed for sample %d: %s", idx_val, exc)

        return rows

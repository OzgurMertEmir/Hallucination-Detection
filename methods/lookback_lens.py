# methods/lookback_lens.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import logging

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from methods.method_cfg import MethodConfig
from internal_features import InternalExampleState

logger = logging.getLogger(__name__)


@dataclass
class LookbackLensConfig(MethodConfig):
    """
    Compute Lookback Lens features (no training), as defined in the paper:
      LR_t^{l,h} = A_t^{l,h}(context) / (A_t^{l,h}(context) + A_t^{l,h}(new))

    The final feature vector is the average of LR_t^{l,h} over the chosen span/window,
    flattened across layers and heads.
    """
    # Aggregate over the full answer ("span") or the last W tokens ("sliding")
    mode: str = "span"              # {"span","sliding"}
    sliding_window: int = 8         # used when mode == "sliding"
    last_k_layers: Optional[int] = None  # if set, only use last K layers
    eps: float = 1e-12              # numeric stability
    return_series: bool = False     # if True, also return the per-step LR series (L, H, a_len)

    def __init__(
        self,
        mode: str = "span",
        window: int = 8,
        last_k_layers: Optional[int] = None,
        eps: float = 1e-12,
        return_series: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.sliding_window = int(window)
        self.last_k_layers = int(last_k_layers) if last_k_layers is not None else None
        self.eps = float(eps)
        self.return_series = bool(return_series)


class LookbackLensFeatures:
    """
    Computes Lookback Lens ratios from self-attention maps you already extract.

    Expects per-example:
      - attentions: List[L] of (H, T, T) attention weight tensors (already softmaxed).
      - q_len, a_len, seq_len: integer lengths.

    Output:
      - "lookback_vector": flattened (L*H,) averaged over span/window
      - "lookback_matrix": (L, H) averaged over span/window
      - optionally "lookback_series": (L, H, a_len) per-step ratios
    """

    def compute(self, st: InternalExampleState, cfg: LookbackLensConfig) -> Dict[str, Any]:
        if torch is None or len(st.attentions) == 0:
            # Minimal fallback: no attentions available
            return {
                "label": int(st.label),
                "lookback_vector": [],
                "lookback_matrix": [],
            }

        attn_per_layer: List[torch.Tensor] = st.attentions  # each (H, T, T), already trimmed to per-example
        L = len(attn_per_layer)
        H = int(attn_per_layer[0].shape[0])
        T = int(st.seq_len)
        N = int(st.q_len)         # number of context tokens
        A_len = int(st.a_len)     # number of generated tokens

        # Stack to (L, H, T, T) for convenience (CPU float32 in your extractor)
        A = torch.stack(attn_per_layer, dim=0)  # (L, H, T, T)

        if A_len <= 0 or T <= 0 or N < 0:
            return {
                "label": int(st.label),
                "lookback_vector": [],
                "lookback_matrix": [],
            }

        # Compute LR^{l,h}_t for each generated token position
        # Global token index for the s-th generated token (1-based) is p = N + s - 1
        lr_series = []
        for s in range(1, A_len + 1):
            p = N + s - 1  # query row index in [0..T-1]

            # For causal attention, the query at p can attend to keys [0..p] (self included).
            # Lookback Lens uses *previously generated* tokens only, so exclude self at p.
            # Context region: [0 .. N-1]; New region: [N .. p-1].
            # Use *averages* (not sums) as in the paper to avoid size bias.
            if N > 0:
                ctx_sum = A[:, :, p, :min(N, p)].sum(dim=-1)            # (L, H)
                ctx_cnt = max(min(N, p), 0)
                ctx_avg = ctx_sum / max(ctx_cnt, 1)
            else:
                ctx_avg = torch.zeros((L, H), dtype=A.dtype)

            new_len = max(p - N, 0)  # number of prior generated tokens available
            if new_len > 0:
                new_sum = A[:, :, p, N:p].sum(dim=-1)                   # (L, H)
                new_avg = new_sum / float(new_len)
            else:
                new_avg = torch.zeros_like(ctx_avg)

            lr = ctx_avg / (ctx_avg + new_avg + cfg.eps)                # (L, H)
            lr_series.append(lr)

        if lr_series:
            LR = torch.stack(lr_series, dim=-1)  # (L, H, A_len)
        else:
            LR = torch.zeros((L, H, 0), dtype=A.dtype)

        # Optionally restrict to the last K layers
        if cfg.last_k_layers is not None and cfg.last_k_layers > 0 and cfg.last_k_layers <= L:
            LR = LR[-cfg.last_k_layers:, :, :]

        # Aggregate over the span/window
        if cfg.mode == "span":
            # average across the entire generated span
            agg = LR.mean(dim=-1) if LR.shape[-1] > 0 else torch.zeros(LR.shape[:2], dtype=A.dtype)
        elif cfg.mode == "sliding":
            w = max(int(cfg.sliding_window), 1)
            if LR.shape[-1] > 0:
                tail = LR[:, :, -w:] if LR.shape[-1] >= w else LR
                agg = tail.mean(dim=-1)  # (L, H)
            else:
                agg = torch.zeros(LR.shape[:2], dtype=A.dtype)
        else:
            raise ValueError(f"Unknown LookbackLens mode: {cfg.mode}")

        lookback_matrix = agg  # (L, H)
        lookback_vector = lookback_matrix.flatten().tolist()

        out: Dict[str, Any] = {
            "label": int(st.label),
            "lookback_vector": lookback_vector,
            "lookback_matrix": lookback_matrix.numpy().tolist(),
        }
        if cfg.return_series:
            # Return per-step ratios as (L, H, A_len)
            out["lookback_series"] = LR.numpy().tolist()
        return out

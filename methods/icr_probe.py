
"""
ICR Probe feature extraction (per-layer JSD between residual update projection and attention).
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import logging

try:
    import torch  # type: ignore
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from methods.method_cfg import MethodConfig
from internal_features import InternalExampleState
from methods.ICR_Probe.src.icr_score import ICRScore

logger = logging.getLogger(__name__)


@dataclass
class ICRProbeConfig(MethodConfig):
    top_k: int = 5

    def __init__(self, top_k: int = 5, max_layers: int | None = None):
        super().__init__()
        self.top_k = int(top_k)


def _to_logits(att: "torch.Tensor", eps: float = 1e-12) -> "torch.Tensor":
    """
    Convert attention probabilities to logits if they look like probs.
    Works for [..., M] or [..., Q, M]. Returns float32 for stability.
    """
    import torch
    att_f = att.float()
    sums = att_f.sum(dim=-1)
    is_prob = (att_f.min() >= 0) and torch.allclose(
        sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3
    )
    return torch.log(att_f.clamp_min(eps)) if is_prob else att_f

def _pack_for_icr(
    hidden_per_layer: List["torch.Tensor"],    # each (T, d), length = L+1
    attn_per_layer: List["torch.Tensor"],      # each (H, T, T), length = L
    q_len: int,
    a_len: int,
    device: str = "cpu",
):
    import torch

    assert len(hidden_per_layer) >= 1 and len(attn_per_layer) >= 1
    T = hidden_per_layer[0].shape[0]
    assert all(h.shape[0] == T for h in hidden_per_layer)
    assert all(a.dim() == 3 and a.shape[1] == T and a.shape[2] == T for a in attn_per_layer)
    assert q_len + a_len == T

    L_plus_1 = len(hidden_per_layer)
    L = len(attn_per_layer)

    # move to device; hidden as float32 for stability
    hidden_per_layer = [h.to(device, dtype=torch.float32) for h in hidden_per_layer]
    attn_per_layer   = [a.to(device) for a in attn_per_layer]

    hidden_steps, attn_steps = [], []

    # ---- step 0: prompt block ----
    hs_step0 = [hidden_per_layer[l][:q_len, :].unsqueeze(0) for l in range(L_plus_1)]  # (1, q_len, d)
    attn_step0 = []
    for l in range(L):
        a = attn_per_layer[l][:, :q_len, :q_len]      # (H, q_len, q_len)
        a = _to_logits(a)                             # -> float32
        attn_step0.append(a.unsqueeze(0))             # (1, H, q_len, q_len)
    hidden_steps.append(hs_step0)
    attn_steps.append(attn_step0)

    # ---- steps 1..a_len ----
    for s in range(1, a_len + 1):
        cur_len = q_len + s

        # hidden: the newly generated token
        hs_step_s = [hidden_per_layer[l][cur_len-1:cur_len, :].unsqueeze(0)    # (1, 1, d)
                     for l in range(L_plus_1)]

        # attention: **include self** so the row has length cur_len
        attn_step_s = []
        for l in range(L):
            row = attn_per_layer[l][:, cur_len-1, :cur_len]     # (H, cur_len)  <-- include self
            row = _to_logits(row)                               # float32
            attn_step_s.append(row.unsqueeze(0).unsqueeze(2))   # (1, H, 1, cur_len)

        hidden_steps.append(hs_step_s)
        attn_steps.append(attn_step_s)

    core_positions = {
        "user_prompt_start": 0,
        "user_prompt_end": q_len,
        "response_start": q_len,
    }
    return hidden_steps, attn_steps, core_positions
    
class ICRProbeFeatures:
    def __init__(self, device: Union[str, "torch.device"] = "cpu"):
        self.device = str(device) if device is not None else "cpu"

    def compute(self, internal_state: InternalExampleState, cfg: ICRProbeConfig) -> Dict[str, object]:
        if torch is None:
            return {"label": int(internal_state.label)}

        icr_device = self.device or "cpu"
        if icr_device.startswith("cuda"):
            if not torch.cuda.is_available():
                icr_device = "cpu"
        elif torch.cuda.is_available():
            # Fall back to the current CUDA context when running on GPU without an explicit string.
            icr_device = f"cuda:{torch.cuda.current_device()}"

        hidden_steps, attn_steps, core_positions = _pack_for_icr(
            hidden_per_layer=internal_state.hidden_states,    # list of (T, d), L+1
            attn_per_layer=internal_state.attentions,         # list of (H, T, T), L
            q_len=internal_state.q_len,
            a_len=internal_state.a_len,
            device=icr_device,
        )

        icr_calculator = ICRScore(
            hidden_states=hidden_steps,
            attentions=attn_steps,
            skew_threshold=0,
            entropy_threshold=1e5,
            core_positions=core_positions,
            icr_device=icr_device,
        )

        # Compute ICR scores with config
        icr_scores, top_p_mean = icr_calculator.compute_icr(
            top_k=cfg.top_k,
            top_p=0.1, 
            pooling='mean',
            attention_uniform=False,
            hidden_uniform=False,
            use_induction_head=True
        )

        return {
            "qid": internal_state.qid,
            "label": int(internal_state.label),
            "icr_scores": icr_scores,
            "top_p_mean": top_p_mean,
        }

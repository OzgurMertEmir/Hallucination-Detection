# methods/lap_eigvals.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import logging

try:
    import torch
    from torch import Tensor
except Exception:
    torch = None
    Tensor = Any

from .lapeigvals.hallucinations.features.laplacian import laplacian_diagonal_from_attn  
from .method_cfg import MethodConfig
from ..internal_features import InternalExampleState

logger = logging.getLogger(__name__)


@dataclass
class LapEigvalsConfig(MethodConfig):
    """
    Paper-aligned configuration:
      • Take top-k per (layer, head) from the Laplacian diagonal (per-token),
        concatenate across all layers/heads → z ∈ R^{L·H·k}.

    Parameters
    ----------
    top_k : int
        Number of largest values per (layer, head).
    last_k_layers : Optional[int]
        If set, restrict to the last K layers.
    """
    top_k: int = 10
    last_k_layers: Optional[int] = None
    dtype: str = "float32"

    def __init__(
        self,
        top_k: int = 10,
        last_k_layers: Optional[int] = None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.top_k = int(top_k)
        self.last_k_layers = int(last_k_layers) if last_k_layers is not None else None
        self.dtype = dtype


class LapEigvalsFeatures:
    """
    Minimal, paper-aligned features using the official repo function.

    Returns
    -------
    dict with:
      - "label": int
      - "lap_eigvals_vector": List[float] length L*H*k
      - optional "lap_eigvals_per_token": List[L][H][T]
    """

    def compute(self, st: InternalExampleState, cfg: LapEigvalsConfig) -> Dict[str, Any]:
        if torch is None or not st.attentions:
            return {
                "label": int(st.label),
                "lap_eigvals_vector": [],
            }

        # 1) select layers (optionally last K)
        attn_layers: List[Tensor] = st.attentions
        if cfg.last_k_layers and cfg.last_k_layers > 0:
            attn_layers = attn_layers[-cfg.last_k_layers :]

        # ensure dtype/device consistency
        first = attn_layers[0]
        device = first.device
        if device.type == "cuda":
            target_dtype = torch.float16
        else:
            target_dtype = getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else first.dtype
        attn_layers = [A.to(dtype=target_dtype, device=device) for A in attn_layers]

        # 2) official Laplacian diagonal per (L, H, T)
        #    (eigenvalues of the lower-triangular Laplacian equal its diagonal; repo returns it per token)
        lap_diag = laplacian_diagonal_from_attn(attn_layers)  # (L, H, T)
        if isinstance(lap_diag, list):
            lap_diag = torch.stack(lap_diag, dim=0)
        if lap_diag.dim() != 3:
            raise RuntimeError(f"Expected (L,H,T); got shape {tuple(lap_diag.shape)}")

        L, H, T = lap_diag.shape
        k_eff = max(1, min(int(cfg.top_k), T))

        # 3) top-k per head per layer (across tokens), then concat → z ∈ R^{L·H·k}
        z: List[float] = []
        for l in range(L):
            values, _ = torch.topk(
                lap_diag[l],
                k=k_eff,
                dim=-1,
                largest=True,
                sorted=True,
            )  # (H, k)
            z.extend(values.to(dtype=torch.float32).cpu().reshape(-1).tolist())

        out: Dict[str, Any] = {
            "qid": st.qid,
            "label": int(st.label),
            "lap_eigvals_vector": z,
        }

        return out

    def compute_batch(self, states: List[InternalExampleState], cfg: LapEigvalsConfig) -> List[Dict[str, Any]]:
        return [self.compute(st, cfg) for st in states]

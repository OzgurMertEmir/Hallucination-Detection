
"""
FactCheckmate feature extraction (paper-faithful).

- Pool FFN (MLP) activations over *prefix (prompt)* tokens only.
- Return the pooled vector to be used by a downstream classifier.

This module only *collects features*; it does not train the classifier.
"""
from __future__ import annotations
from typing import Dict, Optional
import logging

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from methods.method_cfg import MethodConfig
from internal_features import InternalExampleState

logger = logging.getLogger(__name__)

class FactCheckmateFeatures:
    def compute(self, internal_state: InternalExampleState, cfg: "FactCheckmateConfig") -> Dict[str, object]:
        """
        Extract a pooled FFN vector.

        Returns
        -------
        dict with:
          - 'fcm_pooled_ffn' : torch.FloatTensor of shape (d,)
          - 'label'          : int (0/1)
        """
        if torch is None:
            return {"label": int(internal_state.label), "fcm_pooled_ffn": None}

        h = internal_state.ffn_hidden  # (T, d)
        if h is None or not isinstance(h, torch.Tensor) or h.numel() == 0:
            return {"label": int(internal_state.label)}

        v = h.mean(dim=0, dtype=torch.float32)  # (d,)
        return {
            "qid": internal_state.qid,
            "label": int(internal_state.label),
            "fcm_pooled_ffn": v.detach(),
        }

    def compute_batch(self, internal_states, cfg: "FactCheckmateConfig"):
        if torch is None:
            return [{"label": int(st.label), "fcm_pooled_ffn": None} for st in internal_states]

        rows = [{"qid":st.qid, "label": int(st.label)} for st in internal_states]
        pooled_vectors = []
        valid_indices = []
        for idx, st in enumerate(internal_states):
            h = getattr(st, "ffn_hidden", None)
            if h is None or not isinstance(h, torch.Tensor) or h.numel() == 0:
                continue
            pooled_vectors.append(h.mean(dim=0, dtype=torch.float32))
            valid_indices.append(idx)

        if pooled_vectors:
            stacked = torch.stack(pooled_vectors, dim=0)
            for offset, idx in enumerate(valid_indices):
                rows[idx]["fcm_pooled_ffn"] = stacked[offset].detach()

        return rows


class FactCheckmateConfig(MethodConfig):
    def __init__(self):
        super().__init__()

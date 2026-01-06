
from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
except Exception:
    torch = None

import pandas as pd

from .internal_features import InternalExampleState
from .methods import (
    FactCheckmateFeatures, FactCheckmateConfig,
    LLMCheckFeatures, LLMCheckConfig,
    ICRProbeFeatures, ICRProbeConfig,
    LapEigvalsFeatures, LapEigvalsConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class CollatedMethodFeatures:
    """
    A tensor-safe container for one method's features.

    - scalars: a pandas DataFrame of true scalars (float/int)
    - tensors: dict[name -> torch.Tensor], each stacked along dim 0 (num_examples, ...)
    - meta:   optional DataFrame of metadata columns (e.g., label)
    """
    scalars: pd.DataFrame
    tensors: Dict[str, "torch.Tensor"]
    meta: pd.DataFrame


class FeatureExtractor:
    """Compute and store features from multiple extraction methods (tensor-safe)."""

    def __init__(
        self,
        device: str,
        method_parameters: Dict[str, Any],
        use_methods: Optional[List[str]] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        stream_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_workers = max(1, num_workers or (os.cpu_count() or 1))
        self.stream_dir = stream_dir
        if self.stream_dir is not None:
            os.makedirs(self.stream_dir, exist_ok=True)
        self._stream_paths: Dict[str, List[str]] = {}
        self._stream_counters: Dict[str, int] = {}
        self._stream_counts: Dict[str, int] = {}

        # Method registry
        self.methods = {
            "factcheckmate": FactCheckmateFeatures(),
            "llm_check": LLMCheckFeatures(),
            "icr_probe": ICRProbeFeatures(device=device),
            "lap_eigvals": LapEigvalsFeatures(),
        }
        if use_methods is not None:
            self.methods = {k: v for k, v in self.methods.items() if k in use_methods}

        # Robust parameter mapping (accept common typos / legacy keys)
        def _get(mp: Dict[str, Any], *names: str, default=None):
            for n in names:
                if n in mp:
                    return mp[n]
            return default

        self.configs = {
            "factcheckmate": FactCheckmateConfig(),
            "llm_check": LLMCheckConfig(
                entropy_top_k=_get(method_parameters, "llmc_entropy_top_k", default=50),
                window_size=_get(method_parameters, "llmc_window_size", default=1),
                answer_only=_get(method_parameters, "llmc_answer_only", default=True),
                hidden_layer_idx=_get(method_parameters, "llmc_hidden_layer_idx", default=None),
                attn_layer_idx=_get(method_parameters, "llmc_attn_layer_idx", default=None),
            ),
            "llm_check": LLMCheckConfig(
                entropy_top_k=_get(method_parameters, "llmc_entropy_top_k", default=50),
            ),
            "icr_probe": ICRProbeConfig(
                top_k=_get(method_parameters, "icr_top_k", default=5),
            ),
            "lap_eigvals": LapEigvalsConfig(
                top_k=_get(method_parameters, "lap_eig_top_k", default=5),
            ),
        }

        # Raw per-example rows per method (dicts of scalars/tensors)
        self._rows: Dict[str, List[Dict[str, Any]]] = {m: [] for m in self.methods.keys()}
        for m in self.methods.keys():
            self._stream_paths[m] = []
            self._stream_counters[m] = 0
            self._stream_counts[m] = 0

    # === Public API ==========================================================

    def extract(self, internal_states: List[InternalExampleState]) -> None:
        """Compute and store method features for a batch of InternalExampleState."""
        if not internal_states:
            return

        worker_count = min(self.max_workers, len(internal_states))
        executor: Optional[ThreadPoolExecutor] = None
        if worker_count > 1:
            executor = ThreadPoolExecutor(max_workers=worker_count)

        try:
            for name, method in self.methods.items():
                cfg = self.configs[name]

                if hasattr(method, "compute_batch"):
                    try:
                        batch_rows = method.compute_batch(internal_states, cfg)  # type: ignore[attr-defined]
                    except Exception as exc:
                        logger.exception("Method %s batch-compute failed; skipping batch. Error: %s", name, exc)
                        batch_rows = []
                    if batch_rows:
                        if self.stream_dir:
                            self._write_stream(name, batch_rows)
                        else:
                            for row in batch_rows:
                                if row and isinstance(row, dict):
                                    self._rows[name].append(row)
                    continue

                def _compute(state: InternalExampleState):
                    try:
                        row = method.compute(state, cfg)
                        if row is None or not isinstance(row, dict):
                            if row is not None and not isinstance(row, dict):
                                logger.warning("Method %s returned non-dict; skipping", name)
                            return None
                        return row
                    except Exception as exc:
                        logger.exception("Method %s failed on example; skipping. Error: %s", name, exc)
                        return None

                if executor is None:
                    rows = [_compute(state) for state in internal_states]
                else:
                    rows = list(executor.map(_compute, internal_states))

                if self.stream_dir:
                    self._write_stream(name, rows)
                else:
                    for row in rows:
                        if row:
                            self._rows[name].append(row)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def get_features(self) -> Dict[str, CollatedMethodFeatures]:
        """
        Return a tensor-safe feature bundle per method.

        For each method:
        - Scalars (int/float or 0-dim tensors) collated into a DataFrame.
        - Tensors with >=1 dim are stacked with torch.stack (dim=0).
        - 'label' or 'isHallucination' are put into meta.
        """
        out: Dict[str, CollatedMethodFeatures] = {}
        for name in self.methods.keys():
            rows = list(self._iter_rows(name))
            if not rows:
                out[name] = CollatedMethodFeatures(
                    scalars=pd.DataFrame(), tensors={}, meta=pd.DataFrame()
                )
                continue

            scalar_cols: Dict[str, List[float]] = {}
            meta_cols: Dict[str, List[Any]] = {}
            tensor_cols: Dict[str, List["torch.Tensor"]] = {}

            for r in rows:
                for k, v in r.items():
                    # Normalize simple tensor scalars
                    if torch is not None and isinstance(v, torch.Tensor):
                        v = v.detach()
                        if v.ndim == 0:
                            v = float(v.item())
                        elif v.ndim >= 1:
                            tensor_cols.setdefault(k, []).append(v)
                            continue

                    # Scalars and meta
                    if k in ("label", "isHallucination", "example_id"):
                        meta_cols.setdefault(k, []).append(v)
                    elif isinstance(v, (int, float)):
                        scalar_cols.setdefault(k, []).append(float(v))
                    else:
                        # Try list/tuple of numbers
                        if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
                            tensor_cols.setdefault(k, []).append(torch.tensor(v, dtype=torch.float32))
                        else:
                            # Fallback: store as meta text
                            meta_cols.setdefault(k, []).append(v)

            # Stack tensors by key (pad 1D if variable length)
            stacked: Dict[str, "torch.Tensor"] = {}
            for k, seq in tensor_cols.items():
                if not seq:
                    continue
                # If shapes are identical, stack directly
                shapes = {tuple(t.shape) for t in seq}
                if len(shapes) == 1:
                    stacked_tensor = torch.stack(seq, dim=0).to(dtype=torch.float32)
                    stacked[k] = stacked_tensor.cpu()
                else:
                    # Only support ragged 1-D padding here
                    if all(t.ndim == 1 for t in seq):
                        maxlen = max(t.numel() for t in seq)
                        pad = []
                        for t in seq:
                            if t.numel() < maxlen:
                                p = torch.nn.functional.pad(t, (0, maxlen - t.numel()))
                                pad.append(p)
                            else:
                                pad.append(t)
                        stacked_tensor = torch.stack(pad, dim=0).to(dtype=torch.float32)
                        stacked[k] = stacked_tensor.cpu()
                    else:
                        # Store as object via pickling if truly ragged 2D+
                        logger.warning("Ragged tensors for key %s – storing as object in meta", k)
                        meta_cols[k] = seq  # type: ignore

            scalars_df = pd.DataFrame(scalar_cols)
            meta_df = pd.DataFrame(meta_cols)
            out[name] = CollatedMethodFeatures(scalars=scalars_df, tensors=stacked, meta=meta_df)
        return out

    def validate_features(self) -> Dict[str, Any]:
        """Lightweight stats for quick smoke checks."""
        summary: Dict[str, Any] = {}
        for method in self.methods.keys():
            if self.stream_dir:
                method_info: Dict[str, Any] = {
                    "num_examples": self._stream_counts[method],
                    "stream_batches": len(self._stream_paths[method]),
                }
                summary[method] = method_info
                continue
            rows = self._rows[method]
            method_info: Dict[str, Any] = {"num_examples": len(rows)}
            if rows:
                # Scalar stats only
                sample = rows[0]
                stats: Dict[str, Dict[str, float]] = {}
                for k in sample.keys():
                    vals: List[float] = []
                    for r in rows:
                        v = r.get(k, None)
                        if v is None:
                            continue
                        if torch is not None and isinstance(v, torch.Tensor) and v.ndim == 0:
                            v = float(v.item())
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
                    if vals:
                        stats[k] = {
                            "min": float(min(vals)),
                            "max": float(max(vals)),
                            "mean": float(sum(vals) / len(vals)),
                        }
                method_info["scalar_stats"] = stats
            summary[method] = method_info
        return summary

    def _write_stream(self, method: str, rows: List[Optional[Dict[str, Any]]]) -> None:
        if not self.stream_dir:
            return
        cleaned: List[Dict[str, Any]] = []
        for row in rows:
            if not row or not isinstance(row, dict):
                continue
            new_row: Dict[str, Any] = {}
            for k, v in row.items():
                if torch is not None and isinstance(v, torch.Tensor):
                    new_row[k] = v.detach().cpu()
                else:
                    new_row[k] = v
            cleaned.append(new_row)
        if not cleaned:
            return
        path = os.path.join(
            self.stream_dir,
            f"{method}_batch{self._stream_counters[method]:05d}.pkl",
        )
        with open(path, "wb") as f:
            pickle.dump(cleaned, f)
        self._stream_counters[method] += 1
        self._stream_paths[method].append(path)
        self._stream_counts[method] += len(cleaned)

    def _iter_rows(self, method: str):
        if self.stream_dir:
            for path in self._stream_paths[method]:
                with open(path, "rb") as f:
                    batch_rows = pickle.load(f)
                for row in batch_rows:
                    yield row
        else:
            for row in self._rows[method]:
                yield row

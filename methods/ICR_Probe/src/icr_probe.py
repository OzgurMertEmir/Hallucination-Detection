# -*- coding: utf-8 -*-
import os
import json
import logging
from typing import List, Dict, Union, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, auc, roc_curve

from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import ICRProbe  # MLP classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Data helpers (repo-internal)
# -----------------------------
def _to_list_of_arrays(x: Any) -> List[np.ndarray]:
    """
    Normalize 'icr_scores' into List[np.ndarray] where each arr has shape [T_i, D]
    Supports:
      - numpy 'object' arrays (ragged)
      - python lists-of-lists
      - dense 3D arrays [N, T, D]
      - 2D arrays [N, D] (already aggregated)
    """
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [np.asarray(a, dtype=np.float32) for a in x.tolist()]
    if isinstance(x, np.ndarray) and x.ndim == 3:
        return [x[i].astype(np.float32) for i in range(x.shape[0])]
    if isinstance(x, np.ndarray) and x.ndim == 2:
        # Already aggregated [N, D] -> treat as T=1
        return [x[i].astype(np.float32)[None, :] for i in range(x.shape[0])]
    if isinstance(x, list):
        return [np.asarray(a, dtype=np.float32) for a in x]
    raise ValueError(f"Unsupported icr_scores container: type={type(x)}; "
                     f"dtype={getattr(x, 'dtype', None)}; ndim={getattr(x,'ndim',None)}")

def _load_file_any(path: str) -> Tuple[List[np.ndarray], np.ndarray, List[Any]]:
    """
    Load {icr_scores, labels, ids?} from .npz/.pkl/.jsonl
    Returns: (scores_list, labels[int N], ids[list])
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npz":
        data = np.load(str(p), allow_pickle=True)
        scores = _to_list_of_arrays(data["icr_scores"])
        labels = np.asarray(data["labels"]).astype(int)
        ids = data["ids"].tolist() if "ids" in data else list(range(len(scores)))
        return scores, labels, ids
    if ext == ".pkl":
        import pickle
        with open(p, "rb") as f:
            obj = pickle.load(f)
        scores = _to_list_of_arrays(obj["icr_scores"])
        labels = np.asarray(obj["labels"]).astype(int)
        ids = obj.get("ids", list(range(len(scores))))
        return scores, labels, ids
    if ext == ".jsonl":
        scores, labels, ids = [], [], []
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                scores.append(np.asarray(rec["icr_scores"], dtype=np.float32))
                labels.append(int(rec["label"]))
                ids.append(rec.get("id", i))
        return scores, np.asarray(labels, dtype=int), ids
    raise ValueError(f"Unsupported file type: {ext}")

def _fit_scaler(train_scores: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean/std across *all time steps* in the train set.
    Returns: (mean[D], std[D])
    """
    # Concatenate along time: [sum(T_i), D]
    cat = np.concatenate(train_scores, axis=0)
    mean = cat.mean(axis=0).astype(np.float32)
    std = cat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)  # guard
    return mean, std

def _apply_scaler(scores: List[np.ndarray], mean: np.ndarray, std: np.ndarray) -> List[np.ndarray]:
    return [(a - mean) / std for a in scores]

def _aggregate(arr: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Aggregate a sequence [T, D] -> fixed vector [D] or [2D] etc.
    Modes:
      - "mean":           mean over time -> [D]
      - "mean_std":       [mean, std]    -> [2D]
      - "mean_max":       [mean, max]    -> [2D]
    """
    assert arr.ndim == 2, f"expected [T, D], got {arr.shape}"
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "mean_std":
        return np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)
    if mode == "mean_max":
        return np.concatenate([arr.mean(axis=0), arr.max(axis=0)], axis=0)
    raise ValueError(f"Unknown pooling mode: {mode}")

class _ICRSimpleDataset(Dataset):
    """Yields (x[D’], y) after temporal pooling + standardization has been applied."""
    def __init__(self, pooled_vectors: np.ndarray, labels: np.ndarray):
        assert pooled_vectors.shape[0] == labels.shape[0]
        self.X = torch.from_numpy(pooled_vectors.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# -----------------------------
# Trainer
# -----------------------------
class ICRProbeTrainer:
    """Trainer class for ICR Probe."""

    def __init__(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        config=None,
        model: Optional[nn.Module] = None,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model  # built later if None
        self.train_loader = train_loader
        self.val_loader = val_loader

        # AMP / scaler
        self.use_amp = bool(getattr(self.config, "use_amp", True)) and self.device.type == "cuda"
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = torch.amp.GradScaler(device=device_type, enabled=(self.use_amp and device_type == 'cuda'))

        # Will be filled in setup_model
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # Early stopping
        self._early_best = float("inf")
        self._early_wait = 0

        # where to save artifacts
        self.save_dir = Path(getattr(self.config, "save_dir", "artifacts/icr_probe"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Public API
    # -------------------------
    def setup_data(self):
        """
        If loaders were provided to __init__, we keep them.
        Otherwise, we load from disk using config (train_file/val_file or data_dir).
        """
        if self.train_loader is not None and self.val_loader is not None:
            logger.info("setup_data: using externally provided DataLoaders.")
            return

        logger.info("setup_data: building DataLoaders from files…")
        data = self._load_data()
        self.train_loader, self.val_loader = self._create_data_loaders(data)

    def setup_model(self):
        """Setup model and optimization components."""
        # Peek a single batch to infer input dimension
        xb, yb = next(iter(self.train_loader))
        input_dim = int(xb.shape[1])

        if self.model is None:
            self.model = ICRProbe(input_dim=input_dim).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        lr = float(getattr(self.config, "learning_rate", 1e-3))
        wd = float(getattr(self.config, "weight_decay", 1e-5))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        # LR scheduler on val loss
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(getattr(self.config, "lr_factor", 0.5)),
            patience=int(getattr(self.config, "lr_patience", 5)),
        )

        # Log model size
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model input_dim={input_dim}, params={n_params/1e6:.2f}M")

    def train(self):
        """Train the model."""
        best_val = float("inf")
        num_epochs = int(getattr(self.config, "num_epochs", 100))
        log_every = int(getattr(self.config, "log_every", 50))

        for epoch in range(num_epochs):
            train_loss = self._train_epoch(log_every=log_every)
            val_metrics = self._validate_epoch()

            # Log metrics
            self._log_metrics(epoch, train_loss, val_metrics)

            # Save best model
            if val_metrics["val_loss"] < best_val:
                best_val = val_metrics["val_loss"]
                self.save_model()
                logger.info("Saved best checkpoint.")

            # LR scheduler step
            self.scheduler.step(val_metrics["val_loss"])

            # Early stopping (on val loss)
            es_patience = int(getattr(self.config, "early_stop_patience", 10))
            min_delta = float(getattr(self.config, "early_stop_min_delta", 0.0))
            improved = (self._early_best - val_metrics["val_loss"]) > min_delta
            if improved:
                self._early_best = val_metrics["val_loss"]
                self._early_wait = 0
            else:
                self._early_wait += 1
                if self._early_wait >= es_patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

    def save_model(self):
        """Save model, configuration, and scaler if available."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_dir / "model.pth")

        # Save config
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save scaler if present
        scaler_path = self.save_dir / "icr_scaler.npz"
        if hasattr(self, "_scaler_mean") and hasattr(self, "_scaler_std"):
            np.savez(scaler_path, mean=self._scaler_mean, std=self._scaler_std)
            logger.info(f"Saved scaler to {scaler_path}")

    # -------------------------
    # Private: epochs
    # -------------------------
    def _train_epoch(self, log_every: int = 50) -> float:
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        clip = float(getattr(self.config, "grad_clip_norm", 1.0))
        for step, (X_batch, y_batch) in enumerate(self.train_loader):
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                logits = self.model(X_batch)            # [B, 1] logits
                loss = self.criterion(logits, y_batch.unsqueeze(1).float())

            self.scaler.scale(loss).backward()
            if clip and clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_steps += 1

            if (step + 1) % log_every == 0:
                logger.info(f"  step {step+1}: train_loss={total_loss/total_steps:.4f}")

        return total_loss / max(total_steps, 1)

    def _validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        metrics: Dict[str, float] = {}
        val_losses: List[float] = []
        val_preds_bin: List[int] = []
        val_probs: List[float] = []
        val_labels: List[int] = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                logits = self.model(X_batch)  # [B,1]
                loss = self.criterion(logits, y_batch.unsqueeze(1).float())
                val_losses.append(float(loss.item()))

                probs = torch.sigmoid(logits).squeeze(1)  # -> [B] in [0,1]
                preds = (probs >= float(getattr(self.config, "halu_threshold", 0.5))).long()

                val_probs.extend(probs.detach().cpu().tolist())
                val_preds_bin.extend(preds.detach().cpu().tolist())
                val_labels.extend(y_batch.detach().cpu().tolist())

        # Metrics
        y_true = np.asarray(val_labels, dtype=int)
        y_prob = np.asarray(val_probs, dtype=np.float32)
        y_pred = np.asarray(val_preds_bin, dtype=int)

        TP = int(((y_pred == 1) & (y_true == 1)).sum())
        FP = int(((y_pred == 1) & (y_true == 0)).sum())
        FN = int(((y_pred == 0) & (y_true == 1)).sum())
        TN = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy = (TP + TN) / max((TP + TN + FP + FN), 1)
        f1 = f1_score(y_true, y_pred) if (y_true.size > 0) else 0.0

        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            optimal_idx = int(np.argmax(tpr - fpr))
            optimal_threshold = float(thresholds[optimal_idx])
        except Exception:
            roc_auc = 0.5
            optimal_threshold = 0.5

        pcc = float(np.corrcoef(y_true, y_prob)[0, 1]) if y_true.size > 1 else 0.0
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        metrics["val_loss"] = avg_val_loss
        metrics["Precision"] = float(precision)
        metrics["Recall"] = float(recall)
        metrics["Accuracy"] = float(accuracy)
        metrics["F1 Score"] = float(f1)
        metrics["ROC-AUC"] = float(roc_auc)
        metrics["PCC"] = float(pcc)
        metrics["optimal_threshold"] = float(optimal_threshold)
        metrics["mean_neg_pred"] = float(y_prob[y_true == 0].mean()) if (y_true == 0).any() else 0.0
        metrics["mean_pos_pred"] = float(y_prob[y_true == 1].mean()) if (y_true == 1).any() else 0.0

        return metrics

    def _log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict):
        """Log training and validation metrics."""
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        for name, value in val_metrics.items():
            logger.info(f"{name}: {value:.4f}")

    # -------------------------
    # Private: data pipeline
    # -------------------------
    def _load_data(self) -> Dict[str, Any]:
        """
        Load scores+labels from:
          - explicit files: config.train_file / config.val_file, OR
          - data_dir containing: train.npz and val.npz, OR
          - data_dir containing: icr_scores.npz (will stratify split)
        The file(s) must store keys: "icr_scores" and "labels".
        """
        data = {}
        # explicit files
        train_file = getattr(self.config, "train_file", None)
        val_file = getattr(self.config, "val_file", None)
        data_dir = getattr(self.config, "data_dir", None)

        if train_file:
            logger.info(f"Loading train file: {train_file}")
            tr_scores, tr_labels, _ = _load_file_any(train_file)
            data["train"] = {"scores": tr_scores, "labels": tr_labels}

        if val_file:
            logger.info(f"Loading val file: {val_file}")
            va_scores, va_labels, _ = _load_file_any(val_file)
            data["val"] = {"scores": va_scores, "labels": va_labels}

        if "train" in data and "val" in data:
            return data

        # Fallback to data_dir
        if data_dir is None:
            raise ValueError(
                "No DataLoaders provided and no files found. "
                "Specify Config.train_file / val_file, or Config.data_dir."
            )
        data_dir = Path(data_dir)
        tr_npz = data_dir / "train.npz"
        va_npz = data_dir / "val.npz"
        one_npz = data_dir / "icr_scores.npz"

        if tr_npz.exists() and va_npz.exists():
            tr_scores, tr_labels, _ = _load_file_any(str(tr_npz))
            va_scores, va_labels, _ = _load_file_any(str(va_npz))
            data["train"] = {"scores": tr_scores, "labels": tr_labels}
            data["val"] = {"scores": va_scores, "labels": va_labels}
            return data

        if one_npz.exists():
            logger.info("Found icr_scores.npz; performing stratified split.")
            scores, labels, _ = _load_file_any(str(one_npz))
            val_ratio = float(getattr(self.config, "val_split", 0.2))
            rng = np.random.default_rng(int(getattr(self.config, "seed", 17)))
            idx = np.arange(labels.shape[0])
            train_idx, val_idx = [], []
            for c in np.unique(labels):
                cls = idx[labels == c]
                rng.shuffle(cls)
                n_val = max(1, int(round(val_ratio * len(cls))))
                val_idx.append(cls[:n_val])
                train_idx.append(cls[n_val:])
            train_idx = np.concatenate(train_idx)
            val_idx = np.concatenate(val_idx)

            data["train"] = {
                "scores": [scores[i] for i in train_idx],
                "labels": labels[train_idx],
            }
            data["val"] = {
                "scores": [scores[i] for i in val_idx],
                "labels": labels[val_idx],
            }
            return data

        raise FileNotFoundError(f"Could not locate data in {data_dir}")

    def _create_data_loaders(self, data: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """
        Standardize (train-only), aggregate sequences -> fixed vectors, build loaders.
        """
        pooling = getattr(self.config, "pooling", "mean")
        batch_size = int(getattr(self.config, "batch_size", 16))
        num_workers = int(getattr(self.config, "num_workers", 2))
        pin_memory = bool(getattr(self.config, "pin_memory", True))

        tr_scores: List[np.ndarray] = data["train"]["scores"]
        tr_labels: np.ndarray = data["train"]["labels"]
        va_scores: List[np.ndarray] = data["val"]["scores"]
        va_labels: np.ndarray = data["val"]["labels"]

        # standardize (train-only)
        self._scaler_mean, self._scaler_std = _fit_scaler(tr_scores)
        np.savez(self.save_dir / "icr_scaler.npz", mean=self._scaler_mean, std=self._scaler_std)

        tr_scores_std = _apply_scaler(tr_scores, self._scaler_mean, self._scaler_std)
        va_scores_std = _apply_scaler(va_scores, self._scaler_mean, self._scaler_std)

        # aggregate to fixed vectors
        tr_vecs = np.stack([_aggregate(a, pooling) for a in tr_scores_std], axis=0)
        va_vecs = np.stack([_aggregate(a, pooling) for a in va_scores_std], axis=0)

        # datasets
        train_ds = _ICRSimpleDataset(tr_vecs, tr_labels)
        val_ds = _ICRSimpleDataset(va_vecs, va_labels)

        # balanced sampling (class-imbalance friendly)
        use_weighted = bool(getattr(self.config, "dataset_weight", True))
        if use_weighted:
            unique, counts = np.unique(tr_labels, return_counts=True)
            inv = {int(u): 1.0 / float(c) for u, c in zip(unique, counts)}
            weights = torch.tensor([inv[int(y)] for y in tr_labels], dtype=torch.float)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=False
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=False
            )

        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )

        # quick shape log
        logger.info(f"Train set: {len(train_ds)} examples; Val set: {len(val_ds)} examples.")
        logger.info(f"Vector dimension after pooling('{pooling}'): {train_ds.X.shape[1]}")

        return train_loader, val_loader
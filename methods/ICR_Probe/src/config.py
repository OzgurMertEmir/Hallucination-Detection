from dataclasses import dataclass
import argparse
from typing import Optional

@dataclass
class Config:
    """Configuration for ICR Probe training."""

    # ----------------- Model -----------------
    input_dim: int = 32        # will be inferred; kept for reference
    hidden_dim: int = 128

    # ----------------- Training -----------------
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    use_amp: bool = True                 # mixed precision when CUDA
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0
    log_every: int = 50

    # ----------------- LR schedule -----------------
    lr_factor: float = 0.5
    lr_patience: int = 5

    # ----------------- Data -----------------
    data_dir: Optional[str] = None       # dir with train.npz/val.npz or icr_scores.npz
    train_file: Optional[str] = None     # explicit file path alternative
    val_file: Optional[str] = None
    val_split: float = 0.2               # if a single file is given
    dataset_weight: bool = True          # use WeightedRandomSampler
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 17

    # ----------------- Feature pooling -----------------
    pooling: str = "mean"                # "mean" | "mean_std" | "mean_max"

    # ----------------- Threshold for reporting -----------------
    halu_threshold: float = 0.5

    # ----------------- Paths -----------------
    save_dir: str = "artifacts/icr_probe"

    @classmethod
    def from_args(cls):
        """
        Create config from command line arguments.
        NOTE: We use parse_known_args() so this works inside notebooks
        where IPython injects its own arguments.
        """
        parser = argparse.ArgumentParser(add_help=False)
        # Data & paths
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--train_file", type=str, default=None)
        parser.add_argument("--val_file", type=str, default=None)
        parser.add_argument("--save_dir", type=str, default="artifacts/icr_probe")

        # Training
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--num_epochs", type=int, default=100)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--grad_clip_norm", type=float, default=1.0)
        parser.add_argument("--use_amp", type=int, default=1)
        parser.add_argument("--early_stop_patience", type=int, default=10)
        parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
        parser.add_argument("--log_every", type=int, default=50)

        # Scheduler
        parser.add_argument("--lr_factor", type=float, default=0.5)
        parser.add_argument("--lr_patience", type=int, default=5)

        # Data behavior
        parser.add_argument("--val_split", type=float, default=0.2)
        parser.add_argument("--dataset_weight", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--pin_memory", type=int, default=1)
        parser.add_argument("--seed", type=int, default=17)

        # Pooling + threshold
        parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "mean_std", "mean_max"])
        parser.add_argument("--halu_threshold", type=float, default=0.5)

        args, _ = parser.parse_known_args()
        d = vars(args)

        # coerce tiny ints -> bools
        d["use_amp"] = bool(d["use_amp"])
        d["dataset_weight"] = bool(d["dataset_weight"])
        d["pin_memory"] = bool(d["pin_memory"])

        return cls(**d)
"""
Entry point for running the hallucination detection pipeline.

This script loads the HaluEval dataset, generates answers using a
language model, extracts features using multiple detection methods,
creates labels, trains a classifier and reports performance metrics.

Usage example:

```
python main.py --model_name Qwen/Qwen2.5-0.5B --max_examples 500
```
"""
from __future__ import annotations

import argparse
import logging, sys
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yaml, os
import pickle

try:
    import torch
    import os
except Exception:
    torch = None  # type: ignore

from .data import (
    HaluEvalDataset,
    MedMCQADataset,
    MMLUDataset,
    QAExample,
    DATASET_REGISTRY,
)
from .build_features import FeatureBuilder

# === Distributed helper functions ===

def _partition_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Return the [start, end) index range for the given rank."""
    if total <= 0:
        return 0, 0
    chunk = (total + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, total)
    return start, end


def _instantiate_datasets(dataset_specs: List[Tuple[str, Dict[str, Any]]]):
    """Instantiate datasets from lightweight (name, kwargs) specs."""
    datasets = []
    for name, kwargs in dataset_specs:
        dataset_cls = DATASET_REGISTRY.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset '{name}'")
        datasets.append(dataset_cls(**kwargs))
    return datasets


def _load_partition(
    dataset_specs: List[Tuple[str, Dict[str, Any]]],
    max_examples: Optional[int],
    start: int,
    end: int,
) -> List[QAExample]:
    """
    Materialise only the slice of examples assigned to a worker.

    This mirrors the ordering used in the single-process path: iterate each
    dataset in sequence and honour the per-dataset ``max_examples`` cap.
    """
    subset: List[QAExample] = []
    if start >= end:
        return subset

    cursor = 0
    for name, kwargs in dataset_specs:
        dataset_cls = DATASET_REGISTRY.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset '{name}'")
        dataset = dataset_cls(**kwargs)
        for ex in dataset.examples(max_examples=max_examples):
            if cursor >= end:
                break
            if cursor >= start:
                subset.append(ex)
            cursor += 1
        if cursor >= end:
            break
    return subset


def _count_examples(dataset_specs: List[Tuple[str, Dict[str, Any]]], max_examples: Optional[int]) -> int:
    """Count the number of examples that would be processed given the specs."""
    total = 0
    for name, kwargs in dataset_specs:
        dataset_cls = DATASET_REGISTRY.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset '{name}'")
        dataset = dataset_cls(**kwargs)
        length = len(dataset)
        if max_examples is not None:
            total += min(length, max_examples)
        else:
            total += length
    return total


def _distributed_worker(
    rank: int,
    world_size: int,
    dataset_specs: List[Tuple[str, Dict[str, Any]]],
    args: argparse.Namespace,
    method_parameters: Dict[str, Any],
    total_examples: int,
    result_queue,
) -> None:
    """
    Worker function launched in each subprocess when running with multiple GPUs.

    Each worker receives a ``rank`` in ``[0, world_size)`` and processes a slice
    of the input examples on its own GPU.  The results are returned to the
    parent process via ``result_queue``.

    Parameters
    ----------
    rank : int
        Local rank of the worker (0-indexed).
    world_size : int
        Total number of workers launched.  Should match ``args.num_gpus``.
    dataset_specs : List[Tuple[str, Dict[str, Any]]]
        Lightweight dataset descriptions that allow each worker to materialise
        only its assigned slice.
    args : argparse.Namespace
        Command‑line arguments from the main process.  This contains the
        model_name, judge model, max_new_tokens, batch size, etc.
    method_parameters : Dict[str, Any]
        YAML configuration for feature extraction methods.
    total_examples : int
        Total number of examples across all datasets (subject to max_examples).
    result_queue : multiprocessing.Queue
        IPC queue used to send the collated features back to the parent.

    Returns
    -------
    None
    """
    import torch
    import logging
    from .build_features import FeatureBuilder  # type: ignore

    logger = logging.getLogger(f"worker-{rank}")
    logger.setLevel(logging.INFO)

    start, end = _partition_range(total_examples, rank, world_size)
    if start >= end:
        logger.warning("Rank %d: no data assigned", rank)
        result_queue.put((rank, None))
        return

    subset = _load_partition(dataset_specs, args.max_examples, start, end)
    if not subset:
        logger.warning("Rank %d: no data assigned", rank)
        result_queue.put((rank, None))
        return

    # Determine device for this worker: map rank to cuda:{rank} if using CUDA
    if args.device and args.device.startswith("cuda"):
        device = f"cuda:{rank}"
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        device = args.device or "cpu"

    # Build features on the partition
    stream_dir = os.path.join(args.output_dir, f"stream_rank{rank}")
    fb = FeatureBuilder(
        model_name_or_path=args.model_name,
        judge_model_name=args.judge_model_name,
        method_parameters=method_parameters,
        device=device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        stream_dir=stream_dir,
    )
    features = fb.build_features(subset)

    # Persist per-rank features to disk to avoid large IPC payloads.
    rank_path = os.path.join(args.output_dir, f"features_rank{rank}.pkl")
    with open(rank_path, "wb") as f:
        pickle.dump(features, f)

    # Free GPU/CPU memory before signaling completion.
    del features
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = fb.feature_extractor.validate_features()
    result_queue.put((rank, {
        "features_path": rank_path,
        "stream_dir": stream_dir,
        "num_examples": len(subset),
        "summary": summary,
    }))
    logger.info("Rank %d: processed %d examples", rank, len(subset))

def main() -> None:
    parser = argparse.ArgumentParser(description="Hallucination detection pipeline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="HuggingFace model identifier")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4o-mini", help="Open AI model identifier")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the language model on")
    parser.add_argument("--max_examples", type=int, default=500, help="Limit the number of examples (for demonstration)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for answer generation")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum new tokens to generate per answer")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to utilise for distributed inference")
    parser.add_argument("--output_dir", type=str, default="features_output", help="Directory to save distributed features")
    parser.add_argument("--log_level", type=str, default="INFO", help="Python logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    if getattr(args, "num_gpus", 1) > 1:
        if not (args.device and args.device.startswith("cuda")):
            logging.warning(
                "num_gpus set to %d but device '%s' is not a CUDA device; falling back to single process.",
                args.num_gpus,
                args.device,
            )
            args.num_gpus = 1
        elif torch is None or not torch.cuda.is_available():
            logging.warning(
                "num_gpus set to %d but CUDA is not available; falling back to single process.",
                args.num_gpus,
            )
            args.num_gpus = 1

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)

    dataset_specs: List[Tuple[str, Dict[str, Any]]] = []
    dataset_specs.append(("HaluEvalDataset", {"config": "qa"}))
    dataset_specs.append(("HaluEvalDataset", {"config": "dialogue"}))
    dataset_specs.append(("HaluEvalDataset", {"config": "summarization"}))
    dataset_specs.append(("MMLUDataset", {}))
    dataset_specs.append(("PsiloQADataset", {}))
    dataset_specs.append(("DefAnDataset", {}))

    config_path = os.path.join(os.path.dirname(__file__), "method_config.yaml")
    try:
        with open(config_path, "r") as f:
            method_parameters = yaml.load(f, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        logger.error("Configuration file 'method_config.yaml' not found at %s", config_path)
        return

    # If the user requested multiple GPUs, we spawn one worker per GPU.  Each worker
    # loads the model on its assigned device and processes a slice of the dataset.
    if getattr(args, "num_gpus", 1) > 1:
        world_size = args.num_gpus
        total_examples = _count_examples(dataset_specs, args.max_examples)
        if total_examples == 0:
            logger.warning("No examples found for distributed processing; aborting.")
            return

        logger.info("Total examples to distribute: %d", total_examples)

        os.makedirs(args.output_dir, exist_ok=True)
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        mp.spawn(
            _distributed_worker,
            args=(world_size, dataset_specs, args, method_parameters, total_examples, result_queue),
            nprocs=world_size,
            join=True,
        )

        # After all workers finish, gather the per‑rank feature files into a single structure
        from .features import CollatedMethodFeatures  # type: ignore

        per_rank_info: Dict[int, Dict[str, Any]] = {}
        while len(per_rank_info) < world_size:
            rank, payload = result_queue.get()
            per_rank_info[rank] = payload

        all_features: Dict[str, CollatedMethodFeatures] = {}
        for rank, info in sorted(per_rank_info.items()):
            if info is None:
                logger.warning("Rank %d returned no metadata; skipping merge.", rank)
                continue
            path = info.get("features_path") if isinstance(info, dict) else None
            if not path or not os.path.exists(path):
                logger.warning("Rank %d feature file missing at %s; skipping merge.", rank, path)
                continue
            with open(path, "rb") as f:
                part = pickle.load(f)
            if not isinstance(part, dict):
                logger.warning("Rank %d features have unexpected type %s; skipping merge.", rank, type(part))
                continue

            for method_name, collated in part.items():
                if method_name not in all_features:
                    all_features[method_name] = CollatedMethodFeatures(
                        scalars=collated.scalars.copy(),
                        tensors={k: v.clone() for k, v in collated.tensors.items()},
                        meta=collated.meta.copy(),
                    )
                else:
                    all_features[method_name].scalars = pd.concat([
                        all_features[method_name].scalars,
                        collated.scalars,
                    ], ignore_index=True)
                    for k, v in collated.tensors.items():
                        if k in all_features[method_name].tensors:
                            all_features[method_name].tensors[k] = torch.cat(
                                [all_features[method_name].tensors[k], v], dim=0
                            )
                        else:
                            all_features[method_name].tensors[k] = v.clone()
                    all_features[method_name].meta = pd.concat([
                        all_features[method_name].meta,
                        collated.meta,
                    ], ignore_index=True)

            summary = info.get("summary") if isinstance(info, dict) else None
            if summary:
                logger.info("Rank %d summary: %s", rank, summary)

        all_path = os.path.join(args.output_dir, "features_all.pkl")
        with open(all_path, "wb") as f:
            pickle.dump(all_features, f)
        logger.info(
            "Multi‑GPU feature extraction complete. Collated features saved to %s",
            all_path,
        )
        # If needed, return or further process `all_features` here.  Currently we
        # terminate after saving.
        return

    # Single‑GPU or CPU path: run the pipeline as before
    train_datasets = _instantiate_datasets(dataset_specs)
    examples: List[QAExample] = []
    for dataset in train_datasets:
        for ex in dataset.examples(max_examples=args.max_examples):
            examples.append(ex)

    logger.info("Collected %d examples", len(examples))
    logger.info("Extracting features")

    os.makedirs(args.output_dir, exist_ok=True)
    stream_dir = os.path.join(args.output_dir, "stream_main")
    featureBuilder = FeatureBuilder(
        model_name_or_path=args.model_name,
        judge_model_name=args.judge_model_name,
        method_parameters=method_parameters,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        stream_dir=stream_dir,
    )
    features = featureBuilder.build_features(examples)
    all_path = os.path.join(args.output_dir, "features_all.pkl")
    with open(all_path, "wb") as f:
        pickle.dump(features, f)

if __name__ == "__main__":
    main()

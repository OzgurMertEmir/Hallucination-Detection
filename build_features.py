from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import os
from .data import QAExample
from .internal_features import InternalStateExtractor
from .features import FeatureExtractor
import torch, logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(
        self,
        model_name_or_path: str,
        judge_model_name: str,
        method_parameters,
        device: str,
        *,
        max_new_tokens: int = 64,
        batch_size: int = 1,
        feature_workers: Optional[int] = None,
        capture_dtype: Optional[str] = None,
        stream_dir: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for internal state extraction but is not installed.")
        if capture_dtype is None:
            if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
                capture_dtype = "float16"
            else:
                capture_dtype = "float32"
        self.stream_dir = stream_dir
        self.internal_state_extractor = InternalStateExtractor(model_name_or_path=model_name_or_path,
                                                               judge_model_name=judge_model_name,
                                                               device=device,
                                                               max_new_tokens=max_new_tokens,
                                                               capture_dtype=capture_dtype)
        self.feature_extractor = FeatureExtractor(device=device,
                                                  method_parameters=method_parameters,
                                                  use_methods=["icr_probe","lap_eigvals", 
                                                               "llm_check", "factcheckmate"],
                                                  num_workers=self._resolve_workers(feature_workers),
                                                  stream_dir=stream_dir)
        self.batch_size = batch_size

    def _resolve_workers(self, feature_workers: Optional[int]) -> int:
        if feature_workers is not None and feature_workers > 0:
            return feature_workers
        cpu_count = os.cpu_count() or 1
        # Leave one core headroom for data loading and main thread.
        return max(1, cpu_count - 1)

    def build_features(self, dataset: List[QAExample]):
        total = len(dataset)
        if total == 0:
            logger.info("No examples provided; skipping feature extraction.")
            return self.feature_extractor.get_features()

        index = 0
        current_batch = dataset[index:index + self.batch_size]
        index += len(current_batch)
        if not current_batch:
            return self.feature_extractor.get_features()

        with ThreadPoolExecutor(max_workers=1) as prep_pool:
            future = prep_pool.submit(self.internal_state_extractor.prepare, current_batch)
            with tqdm(total=total, desc="Processing Batches", unit="example") as pbar:
                while True:
                    if index < total:
                        next_batch = dataset[index:index + self.batch_size]
                        index += len(next_batch)
                        next_future = prep_pool.submit(self.internal_state_extractor.prepare, next_batch)
                    else:
                        next_batch = None
                        next_future = None

                    prepared = future.result()
                    internal_states = self.internal_state_extractor.extract(current_batch, prepared=prepared)
                    self.feature_extractor.extract(internal_states)

                    for st in internal_states:
                        if hasattr(st, "hidden_states"):
                            st.hidden_states.clear()
                        if hasattr(st, "attentions"):
                            st.attentions.clear()
                        if hasattr(st, "ffn_hidden"):
                            st.ffn_hidden = None
                        if hasattr(st, "logits"):
                            st.logits = None
                    del internal_states
                    import gc; gc.collect()
                    pbar.update(len(current_batch))

                    if next_future is None:
                        break

                    current_batch = next_batch  # type: ignore[assignment]
                    future = next_future

        summary = self.feature_extractor.validate_features()
        logger.info("Extracted features summary:\n\n'%s'", summary)
        return self.feature_extractor.get_features()
            

# Hallucination Detector

Lightweight pipeline for building hallucination-detection features on top of multiple QA-style datasets. It:
- loads several benchmarks (HaluEval `qa`/`dialogue`/`summarization`, MMLU-Pro, PsiloQA, DefAn),
- prompts a Hugging Face causal LM to answer each question,
- optionally labels the answers with an LLM-as-a-judge (via LangChain + OpenAI),
- extracts features from four published methods (FactCheckmate, LLM-Check, ICR Probe, Laplacian Eigenvalues),
- writes a collated feature bundle you can use to train or evaluate your own classifiers.

The current entry point (`main.py`) focuses on feature generation; model training is left to the user.

## Quick start

1) Python environment (3.9+ recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Credentials (optional but recommended)
   - `OPENAI_API_KEY` for judge labeling via `langchain_openai` + `ChatOpenAI`.
   - `HF_HOME` / `HF_TOKEN` if you use a private Hugging Face cache.
3) Run feature extraction (single GPU/CPU)
```bash
python main.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --device cuda \
  --max_examples 200 \
  --output_dir features_output
```
This downloads the datasets, generates answers, extracts features, and saves `features_output/features_all.pkl`.

### Multi-GPU
Set `--num_gpus N` (with `--device cuda`) to shard the work; per-rank files are merged automatically into `features_output/features_all.pkl`.

### CLI flags (high impact)
- `--model_name`: HF causal LM to answer questions.
- `--judge_model_name`: OpenAI model name for the judge (defaults to `gpt-4o-mini`); set `OPENAI_API_KEY`.
- `--device`: `cpu` or `cuda[:idx]`.
- `--max_examples`: cap total examples across datasets.
- `--batch_size`, `--max_new_tokens`: generation controls.
- `--num_gpus`: number of workers for distributed inference.
- `--output_dir`: where feature pickle(s) and streamed batches are written.

## Configuration

Feature-method knobs live in `method_config.yaml`:
- `llmc_entropy_top_k`, `llmc_window_size`, `llmc_answer_only`, `llmc_hidden_layer_idx`, `llmc_attn_layer_idx` (LLM-Check)
- `icr_top_k` (ICR Probe)
- `lap_eig_top_k` (Laplacian Eigenvalues)

Adjust these if you want fewer/more per-layer values or different entropy windows.

## Method references

- FactCheckmate — feature extractor mirrors the paper description; public paper link not provided upstream yet.
- LLM-Check (NeurIPS 2024) — [paper](https://openreview.net/forum?id=LYx4w3CAgy) | [repo](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection)
- ICR Probe (ACL 2025) — [repo](https://github.com/XavierZhang2002/ICR_Probe) (paper forthcoming; abstract in README)
- Spectral Laplacian Eigenvalues (EMNLP 2025) — [paper](https://arxiv.org/abs/2502.17598) | [repo](https://github.com/graphml-lab-pwr/lapeigvals)

## Experiments notebook

- See `notebooks/ensemble_pipelines.ipynb` for end-to-end training/evaluation of the base detectors and a meta-ensemble over their outputs. It:
  - loads collated feature pickles (`features_*.pkl`) and balances/merges datasets,
  - trains per-method classifiers (FactCheckmate, LLM-Check, ICR Probe, Lap Eigvals) and a meta logistic regressor,
  - includes plotting helpers and cross-dataset evaluation snippets.
- The notebook defaults to a Colab-style path (`FeaturePaths.root = "/content/drive/MyDrive/halu_features"`). Set `FeaturePaths.root` to your local features directory (for example `features_output`) before running locally, and remove or adapt Colab-only cells (`pip uninstall`, `!unzip ...`) as needed.
- Open locally with `jupyter lab notebooks/ensemble_pipelines.ipynb` or upload to Colab; ensure `requirements.txt` is installed in your environment.

## Repository layout

```
main.py                 # CLI for dataset loading + feature extraction
build_features.py       # orchestrates generation + feature passes
internal_features.py    # runs the HF model, collects hidden states/attn/logits, optional judge labels
features.py             # collates per-method outputs into tensor-safe bundles
method_config.yaml      # knobs for each detection method
data/                   # dataset wrappers (HaluEval, MMLU-Pro, PsiloQA, DefAn, MedMCQA)
methods/                # feature extractors + vendored method assets
  ├── factcheckmate.py
  ├── llm_check.py
  ├── icr_probe.py
  ├── lap_eigvals.py
  ├── lookback_lens.py
  ├── ICR_Probe/                 # upstream ICR Probe snippet
  ├── LLM_Check_Hallucination_Detection/  # upstream utilities
  └── lapeigvals/                # upstream Laplacian Eigenvalue helpers
notebooks/             # experiment notebook(s)
prompts.py             # prompt templates for answering + judge prompting
labeling_judge.py      # minimal LLM-as-a-judge helper (LangChain UQLM panel)
requirements.txt
```

Outputs from runs are ignored via `.gitignore` (`features_output/`, `.pkl`, caches).

## Usage notes
- Dataset downloads happen on first run via `datasets`; ensure internet access or a local cache.
- Without an OpenAI key, labels fall back to dataset-provided references (where present).
- Some vendored method code expects GPU tensors for best performance; CPU works but is slower.
- If you want to add training scripts, the collated pickle uses the schema in `features.py::CollatedMethodFeatures` (`scalars`, `tensors`, `meta`).

## Troubleshooting
- Import errors for `torch`/`transformers`: install a GPU-specific wheel (`pip install 'torch==<version>+cu121' -f https://download.pytorch.org/whl/torch_stable.html`).
- Judge timeouts: lower `--batch_size` or disable judge by unsetting `OPENAI_API_KEY` (dataset labels will be used when available).
- Slow HF downloads: set `export HF_HUB_ENABLE_HF_TRANSFER=1` for faster transfers.

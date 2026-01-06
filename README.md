# Hallucination Detector

This repository implements an end‑to‑end pipeline for **hallucination detection** in large language models (LLMs).  The goal is to build a lightweight classifier that decides whether an answer produced by an LLM is based on the underlying knowledge or whether the model has *hallucinated* unsupported information.  When a hallucination is likely the detector should abstain from answering.

The pipeline consists of the following stages:

1. **Dataset wrappers** – A small wrapper around the [HaluEval dataset](https://huggingface.co/datasets/pminervini/HaluEval) for loading question‑answer pairs.  The `qa` configuration of this dataset provides fields `knowledge`, `question`, `right_answer` and `hallucinated_answer`【852640091365365†L128-L137】.  We also support the `qa_samples` configuration, which contains an `answer` and a human‑provided `hallucination` label【852640091365365†L146-L153】.
2. **Baseline hallucination detection methods** – Implementations of three recent hallucination detection methods described in the literature:
   - **FactCheckmate** – learns a classifier over internal hidden states of an LLM to pre‑emptively predict hallucinations.  The FactCheckmate paper shows that a simple classifier over aggregated hidden states can predict upcoming hallucinations with over 70 % accuracy【924916950037334†L16-L41】.
   - **LLM‑Check** – analyses internal hidden states, attention maps and output prediction probabilities of a language model to detect hallucinations using a single response.  The authors show that their techniques are extremely compute‑efficient and achieve large speed‑ups over baselines【742629337513176†L16-L50】.
   - **ICR Probe** – introduces the **Information Contribution to Residual Stream (ICR) score**, which quantifies how each network module contributes to the hidden state update.  By tracking how hidden representations evolve across layers the ICR Probe distinguishes hallucinations more reliably than static methods【545815651594640†L45-L56】.
3. **Model inference** – Use a small open‑source LLM (the repository provides configuration for the 0.5 B parameter version of Qwen2.5) to generate answers for each question in the dataset.  Responses are cached locally.
4. **Label generation** – Each question is labelled as hallucinated or not.  When a ground truth `right_answer` is available we compute a token‑level F1 similarity between the model’s answer and the reference answer using the standard SQuAD evaluation procedure【970334575385018†L246-L272】.  The example is marked as correct (label 0) if the F1 exceeds a configurable threshold and is larger than its similarity to the hallucinated answer; otherwise it is marked as hallucinated (label 1).  If only the `hallucination` flag is available it is used directly.
5. **Feature construction** – Each detection method extracts features from the question/answer pair.  FactCheckmate aggregates the model’s hidden states before decoding begins using mean, max and last‑token pooling across tokens and summarizes these pooled representations via their mean, variance and maximum absolute values across layers【924916950037334†L235-L277】.  LLM‑Check computes the *hidden score* (mean log‑determinant of the hidden state covariance matrix) and the *attention score* (mean log‑determinant of the attention kernel) for each layer and summarizes these values across layers; it also includes perplexity and entropy‑based features of the answer【742629337513176†L618-L737】.  The ICR Probe computes an **Information Contribution to Residual Stream** (ICR) score by comparing, for each layer and token, the direction of the hidden state update to the attention weights; this is quantified via the Jensen–Shannon divergence between the update projection distribution and the attention distribution【910855256871511†L301-L444】.  Additional simple lexical features (answer length, digit counts and overlap with the question) are included for robustness.
6. **Detector training** – A classifier (e.g. logistic regression or random forest) is trained on the features and labels.  The detector outputs a probability that an answer is hallucinated.  If the probability exceeds a configurable threshold the detector predicts a hallucination; otherwise it accepts the answer.  An abstention mechanism is built on top of the probabilities by introducing a second threshold: predictions within a grey zone are abstained.  The abstention mechanism is evaluated not only by coverage but also by the reduction in misinformation: we compare the fraction of hallucinations in the accepted answers to the fraction when no abstention is applied.
7. **Evaluation** – The repository contains scripts to train the detector and report accuracy, precision/recall/F1 and abstention statistics on a held‑out portion of the dataset.

## Getting started

1. **Install dependencies**

   The code depends on the following Python libraries:

   ```bash
   pip install torch transformers datasets numpy scikit‑learn pandas tqdm
   ```

   If you only plan to use the pre‑computed features provided in this repository you can omit `torch` and `transformers`.

2. **Running the pipeline**

   ```bash
   python main.py \
       --dataset_config qa \   # or qa_samples
       --model_name Qwen/Qwen2.5-0.5B \
       --max_examples 1000       # limit for demonstration purposes
   ```

    The script downloads the dataset, runs the language model to generate answers, extracts features, trains the detector and prints an evaluation summary.  It also reports the percentage of hallucinated answers before and after applying abstention.

## Repository structure

```
hallucination_detector/
├── README.md               # this file
├── requirements.txt        # pip dependencies
├── halueval_dataset.py     # dataset wrapper around HuggingFace data
├── methods/
│   ├── __init__.py
│   ├── factcheckmate.py    # FactCheckmate feature extractor
│   ├── llm_check.py        # LLM‑Check feature extractor
│   └── icr_probe.py        # ICR Probe feature extractor
├── features.py             # orchestrates feature extraction for all methods
├── detector.py             # training and evaluation of hallucination detector
├── main.py                 # end‑to‑end pipeline CLI
└── utils.py                # helper functions
```

## Notes

* The implementation in this repository follows the high‑level descriptions of the respective papers.  FactCheckmate and the ICR Probe rely on internal representations of the underlying language model; the code uses the HuggingFace `transformers` API to access intermediate hidden states.  In particular, the ICR implementation computes Jensen–Shannon divergences between attention distributions and the directions of hidden state updates across layers【910855256871511†L301-L444】.  When these packages are unavailable, the feature extractors fall back to simple heuristics using only the surface form of the question and answer.
- The HaluEval dataset was created for hallucination benchmarking and contains multiple configurations.  We use the `qa` split for demonstration, which has fields `knowledge`, `question`, `right_answer` and `hallucinated_answer`【852640091365365†L128-L137】.
* FactCheckmate pre‑emptively detects hallucinations by learning a classifier over a model’s hidden states【924916950037334†L16-L24】.  LLM‑Check analyses a single response by examining internal attention maps, hidden activations and output probabilities, achieving compute efficiency【742629337513176†L16-L50】.  The ICR Probe introduces an information contribution score to track how hidden states evolve across layers【545815651594640†L45-L56】 and quantifies this via Jensen–Shannon divergence between update directions and attention distributions【910855256871511†L301-L444】.

For more details please consult the corresponding papers.
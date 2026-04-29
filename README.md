# Pre-training, Fine-tuning, and Prompting Strategies for Bug Fixing

**CSCI 455 — GenAI for Software Development**  
**Name:** Lily Walker 

---

## Overview

This project investigates whether pre-training helps a small language model learn to fix Java bugs. It compares four configurations evaluated on the [CodeXGlue Code Refinement (medium)](https://huggingface.co/datasets/google/code_x_glue_cc_code_refinement) benchmark:

| Configuration | Description |
|---|---|
| **Pipeline A** | T5-small pre-trained on 50K Java methods → fine-tuned on bug fixing |
| **Pipeline B** | T5-small initialized from scratch → fine-tuned directly on bug fixing |
| **Qwen Zero-Shot** | Qwen 2.5-Coder-1.5B prompted with no examples |
| **Qwen RAG (3-shot)** | Qwen 2.5-Coder-1.5B prompted with 3 retrieved bug-fix examples |

---

## Repository Structure

```
.
┌── src
    ├── collators.py
    ├── dataset_utils.py
    ├── models.py
    ├── tokenizer_utils.py
    ├── pretrain.py                     # Holistic pretrain
    ├── finetune.py                     # Holistic finetune
    ├── eval_rag.py                     # Holistic eval and rag
├── 01_tokenizing_and_data_prep.ipynb   # Tokenizer training, model init, pre-training (Pipeline A)
├── 02_finetuning.ipynb                 # Fine-tuning Pipelines A and B
├── 03_eval_and_rag.ipynb               # Evaluation (CodeBLEU + exact match) + RAG system
├── README.md
├── requirements.txt
│
├── java_tokenizer/                     # Saved SentencePiece tokenizer (created by Notebook 1)
├── final_pretrained_model/             # Pre-trained T5 checkpoint (created by Notebook 1)
├── pipeline_a_finetuned/               # Pipeline A fine-tuned model (created by Notebook 2)
├── pipeline_b_finetuned/               # Pipeline B fine-tuned model (created by Notebook 2)
└── eval_results/                       # Saved predictions and results JSON (created by Notebook 3)
    ├── predA_results.json
    └── results_summary.json
```

---

## Requirements

### Hardware
- A CUDA-capable GPU is recommended especially for Notebooks 2 and 3.
- Notebook 3 loads both CodeBERT (768M) and Qwen 2.5-Coder-1.5B simultaneously at points.

### Python
Python 3.10 

### Create Virtual Environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

Each notebook includes its own install cell at the top. To install everything upfront:

```bash
pip install \
  datasets \
  transformers \
  sentencepiece \
  tokenizers \
  accelerate \
  torch \
  faiss-cpu \
  sentence-transformers \
  evaluate \
  sacrebleu \
  tree-sitter==0.21.0 \
  tree-sitter-java==0.20.0 \
  codebleu==0.1.7
```
or 
```bash
pip install -r requirements.txt
```

> **Note:** `codebleu==0.1.7`, `tree-sitter==0.21.0`, and `tree-sitter-java==0.20.0` were pinned to exact version because of conflicting dependencies and a breaking API change

---

## Reproducibility
Run the three notebooks in order. Each notebook saves its outputs to disk so the next notebook can load them.

### Notebook 1 — `01_tokenizing_and_data_prep.ipynb`

- Downloads 50,000 Java methods from [CodeSearchNet](https://huggingface.co/datasets/code_search_net)
- Trains a SentencePiece Unigram tokenizer (vocab size 16,384) with all T5 special tokens and 100 sentinel tokens
- Initializes a T5-small model (~60M parameters) from scratch
- Pre-trains using a span corruption objective (15% corruption rate) for 3 epochs
- Runs sanity checks on the tokenizer and dataset

**Outputs written to:**
- `./java_tokenizer/` — tokenizer files (reused by all subsequent notebooks)
- `./final_pretrained_model/` — pre-trained T5 weights (used by Pipeline A in Notebook 2)
- `./t5_pretrain/` — intermediate Trainer checkpoints

---

### Notebook 2 — `02_finetuning.ipynb`

- Loads the tokenizer from `./java_tokenizer/`
- Downloads the CodeXGlue Code Refinement (medium) dataset (~52K train / 6.5K val / 6.5K test)
- **Pipeline A:** Loads `./final_pretrained_model/` and fine-tunes on bug fixing
- **Pipeline B:** Initializes a fresh T5-small from scratch (same config) and fine-tunes directly
- Both pipelines use identical hyperparameters

**Hyperparameters (both pipelines):**

| Parameter | Value |
|---|---|
| Epochs | 10 (with early stopping) |
| Batch size | 8 |
| Learning rate | 5e-5 |
| Warmup steps | 200 |
| Weight decay | 0.01 |
| Eval/save strategy | Per epoch |
| Best model metric | eval_loss |
| Mixed precision | fp16 (CUDA) |

**Outputs written to:**
- `./pipeline_a_finetuned/` — Pipeline A model + tokenizer
- `./pipeline_b_finetuned/` — Pipeline B model + tokenizer


### Notebook 3 — `03_eval_and_rag.ipynb`
- Loads both fine-tuned models and evaluates on the test set (6,545 samples)
- Builds a CodeBERT + FAISS retrieval index over the 52K training pairs
- Loads Qwen 2.5-Coder-1.5B-Instruct and runs zero-shot and 3-shot RAG inference
- Computes CodeBLEU (n-gram, syntax, dataflow) and exact match for all four configurations
- Saves a results summary to `./eval_results/results_summary.json`

**Retriever design:** CodeBERT (`microsoft/codebert-base`) with mean-pooling over the last hidden state, producing 768-dimensional vectors indexed in a FAISS `IndexFlatL2`.

**Outputs written to:**
- `./eval_results/results_summary.json` — all four configurations, full metric breakdown

### Holistic Reproducibility - `pretrain.py`, `finetune.py`, `eval_rag.py`
```bash
# Pretrain
python pretrain.py

# Finetune
python finetune.py

# Eval & RAG
python eval_rag.py

# result in same output files from the jupyter notebooks
```

---

## Results Summary

| Configuration | Exact Match | CodeBLEU | n-gram | Syntax | Dataflow |
|---|---|---|---|---|---|
| Pipeline A (Pre-trained) | 0.03% | 0.6658 | 0.4558 | 0.7000 | 0.7573 |
| **Pipeline B (From Scratch)** | **0.06%** | **0.7311** | **0.7293** | **0.7125** | **0.7180** |
| Qwen 1.5B Zero-Shot | 0.00% | 0.3574 | 0.0678 | 0.5583 | 0.6857 |
| Qwen 1.5B RAG (3-Shot) | 0.00% | 0.5409 | 0.4283 | 0.6228 | 0.6675 |


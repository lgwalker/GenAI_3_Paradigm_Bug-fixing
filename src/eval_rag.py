import json
import os
import torch
import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)
from codebleu import calc_codebleu
from dataset_utils import get_finetune_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def compute_exact_match(predictions, references):
    """Fraction of predictions that exactly match the reference after stripping whitespace."""
    assert len(predictions) == len(references), "Length mismatch"
    matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return round(matches / len(references), 4)


def compute_codebleu_score(predictions, references, lang="java"):
    """CodeBLEU: code-aware metric combining n-gram, syntax, and dataflow scores."""
    try:
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
        return calc_codebleu(references, predictions, lang=lang, weights=(0.25, 0.25, 0.25, 0.25))
    except Exception as e:
        print(f"CodeBLEU Error: {e}")
        return {
            "codebleu": 0.0,
            "ngram_match_score": 0.0,
            "syntax_match_score": 0.0,
            "dataflow_match_score": 0.0,
        }


def evaluate_model(predictions, references, label):
    """Compute and print CodeBLEU and exact match. Returns a results dict."""
    em = compute_exact_match(predictions, references)
    cb = compute_codebleu_score(predictions, references, lang="java")

    print(f"\n{'-'*50}")
    print(f"Results: {label}")
    print(f"{'-'*50}")
    print(f"  Exact Match  : {em:.4f}  ({em*100:.2f}%)")
    print(f"  CodeBLEU     : {cb['codebleu']:.4f}")
    print(f"    n-gram     : {cb['ngram_match_score']:.4f}")
    print(f"    syntax     : {cb['syntax_match_score']:.4f}")
    print(f"    dataflow   : {cb['dataflow_match_score']:.4f}")

    return {
        "label": label,
        "exact_match": em,
        "codebleu": cb["codebleu"],
        "ngram_match": cb["ngram_match_score"],
        "syntax_match": cb["syntax_match_score"],
        "dataflow_match": cb["dataflow_match_score"],
    }



# T5 Generation


def generate_t5_predictions(model, tokenizer, test_data, batch_size=16, prefix="fix: "):
    """
    Generate predictions from a T5 model on the test set.
    Accepts an already-loaded model (caller is responsible for loading/freeing).
    Returns a list of decoded prediction strings.
    """
    model.eval()
    model.to(DEVICE)
    predictions = []

    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating"):
        batch = test_data[i : i + batch_size]
        inputs = [prefix + b for b in batch["buggy"]]

        encoded = tokenizer(
            inputs,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=256,
                num_beams=4,        # beam search for better output quality
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


# CodeBERT Encoder (shared by RAG index build + test query pre-encoding)

def encode_with_codebert(codebert_model, codebert_tokenizer, code_list, batch_size=64):
    """
    Encode a list of code strings using CodeBERT with mean-pooling.
    Returns a numpy array of shape (len(code_list), 768).
    """
    all_embeddings = []
    codebert_model.eval()

    for i in tqdm(range(0, len(code_list), batch_size), desc="Encoding"):
        batch = code_list[i : i + batch_size]
        encoded = codebert_tokenizer(
            batch,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = codebert_model(**encoded)

        # Mean-pool over token dimension, ignoring padding tokens
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_embeddings = outputs.last_hidden_state
        mean_embeddings = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        all_embeddings.append(mean_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)



# FAISS Retrieval helpers (operate on pre-built index + pre-computed embeddings)


def retrieve_examples_single(index, test_embeddings, train_buggy, train_fixed, query_idx, k=3):
    """
    Retrieve k examples for a single test sample using its pre-computed embedding.
    """
    query_emb = test_embeddings[query_idx : query_idx + 1]
    _, indices = index.search(query_emb.astype(np.float32), k)
    return [(train_buggy[int(i)], train_fixed[int(i)]) for i in indices[0]]


def build_rag_prompt(buggy_code, query_idx, index, test_embeddings, train_buggy, train_fixed, k=3):
    """
    Build a 3-shot RAG prompt using retrieved bug-fix examples.
    Uses pre-computed embedding via query_idx — no model call at inference time.
    """
    examples = retrieve_examples_single(index, test_embeddings, train_buggy, train_fixed, query_idx, k=k)
    prompt = (
        "You are an expert Java developer. Fix the bug in the Java method below.\n"
        "Here are similar bug-fix examples for reference:\n\n"
    )
    for idx, (bug, fix) in enumerate(examples, 1):
        prompt += f"### Example {idx}\nBuggy:\n{bug}\n\nFixed:\n{fix}\n\n"
    prompt += f"### Now fix this method\nBuggy:\n{buggy_code}\n\nFixed:\n"
    return prompt


def build_zeroshot_prompt(buggy_code, query_idx=None):
    """
    Zero-shot prompt — no retrieved examples, just the buggy method.
    query_idx is accepted but unused (keeps signature uniform with build_rag_prompt).
    """
    return (
        "You are an expert Java developer. Fix the bug in the following Java method.\n"
        f"Buggy:\n{buggy_code}\n\nFixed:\n"
    )


# Qwen Batched Generation


def generate_qwen_predictions(qwen_model, qwen_tokenizer, test_data, prompt_fn, label, batch_size=8):
    """
    Generate predictions from Qwen using the given prompt builder function.

    - prompt_fn(buggy_code, query_idx) — RAG variant uses pre-computed embeddings
      so there is no CodeBERT call at inference time.
    - Left-padded tokenizer ensures correct generation for all batch items.
    - Decodes only the newly generated tokens (beyond the input length).

    Returns a list of predicted fixed methods.
    """
    predictions = []
    n = len(test_data)

    for batch_start in tqdm(range(0, n, batch_size), desc=f"Qwen ({label})"):
        batch_indices = range(batch_start, min(batch_start + batch_size, n))

        # Build all prompts for this batch
        prompts = [prompt_fn(test_data[i]["buggy"], i) for i in batch_indices]

        # Tokenize as a left-padded batch
        inputs = qwen_tokenizer(
            prompts,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,       # pads to longest in batch (left side)
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,    # greedy — deterministic
                pad_token_id=qwen_tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens for each item in the batch
        input_len = inputs["input_ids"].shape[1]
        for seq in output_ids:
            new_tokens = seq[input_len:]
            decoded = qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)
            prediction = decoded.strip().split("\n\n")[0].strip()
            predictions.append(prediction)

    return predictions



# Main Execution


if __name__ == "__main__":
    os.makedirs("./eval_results", exist_ok=True)

    print("Loading dataset and tokenizer...")
    dataset = get_finetune_data()
    test_data = dataset["test"]
    references = list(test_data["fixed"])
    print(f"Test samples: {len(test_data)}")

    t5_tokenizer = PreTrainedTokenizerFast.from_pretrained("./java_tokenizer")
    print(f"Tokenizer vocab size: {len(t5_tokenizer)}")

    all_results = []

    # Pipeline A — T5 fine-tuned WITH pre-training

    print("\nLoading Pipeline A model (with pre-training)...")
    model_a = T5ForConditionalGeneration.from_pretrained("./pipeline_a_finetuned")
    preds_a = generate_t5_predictions(model_a, t5_tokenizer, test_data)
    results_a = evaluate_model(preds_a, references, "Pipeline A — With Pre-training")
    all_results.append(results_a)
    del model_a
    torch.cuda.empty_cache()


    # Pipeline B — T5 fine-tuned WITHOUT pre-training

    print("\nLoading Pipeline B model (without pre-training)...")
    model_b = T5ForConditionalGeneration.from_pretrained("./pipeline_b_finetuned")
    preds_b = generate_t5_predictions(model_b, t5_tokenizer, test_data)
    results_b = evaluate_model(preds_b, references, "Pipeline B — Without Pre-training")
    all_results.append(results_b)
    del model_b
    torch.cuda.empty_cache()


    # RAG Setup — CodeBERT encoder + FAISS index
    # Encode train KB and all test queries ONCE upfront, then free CodeBERT.

    print("\nLoading CodeBERT for retrieval...")
    codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert_model = AutoModel.from_pretrained("microsoft/codebert-base").to(DEVICE)

    train_buggy = list(dataset["train"]["buggy"])
    train_fixed = list(dataset["train"]["fixed"])

    print("\nBuilding FAISS index from training knowledge base...")
    kb_embeddings = encode_with_codebert(codebert_model, codebert_tokenizer, train_buggy)
    dimension = kb_embeddings.shape[1]  # 768 for CodeBERT
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(kb_embeddings.astype(np.float32))
    print(f"FAISS index built | Vectors: {faiss_index.ntotal} | Dimension: {dimension}")

    print("\nPre-encoding all test queries with CodeBERT...")
    test_buggy_list = [d["buggy"] for d in test_data]
    test_embeddings = encode_with_codebert(codebert_model, codebert_tokenizer, test_buggy_list)
    print(f"Test queries encoded: {test_embeddings.shape}")

    # Free CodeBERT from GPU — all embeddings are now in numpy arrays
    del codebert_model
    torch.cuda.empty_cache()
    print("CodeBERT freed from GPU.")

    # Bind index + embedding arrays into the prompt functions via lambdas
    def rag_prompt_fn(buggy_code, query_idx):
        return build_rag_prompt(
            buggy_code, query_idx,
            faiss_index, test_embeddings, train_buggy, train_fixed,
            k=3,
        )

    
    # Qwen — load once, run zero-shot then RAG

    print("\nLoading Qwen2.5-Coder-1.5B-Instruct...")
    QWEN_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    qwen_tokenizer.padding_side = "left"   # required for correct batched causal LM generation
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    qwen_model.eval()
    print("Qwen model loaded.")

    # Zero-Shot
    print("\nRunning Qwen Zero-Shot...")
    preds_zeroshot = generate_qwen_predictions(
        qwen_model, qwen_tokenizer, test_data,
        prompt_fn=build_zeroshot_prompt,
        label="zero-shot",
        batch_size=4,
    )
    results_zeroshot = evaluate_model(preds_zeroshot, references, "Qwen 1.5B — Zero-Shot")
    all_results.append(results_zeroshot)

    # RAG (3-shot)
    print("\nRunning Qwen RAG (3-shot)...")
    preds_rag = generate_qwen_predictions(
        qwen_model, qwen_tokenizer, test_data,
        prompt_fn=rag_prompt_fn,
        label="RAG-3shot",
        batch_size=4,
    )
    results_rag = evaluate_model(preds_rag, references, "Qwen 1.5B — RAG 3-shot")
    all_results.append(results_rag)

    del qwen_model
    torch.cuda.empty_cache()

    
    # Final Summary Table

    print("\n" + "-" * 70)
    print("FINAL RESULTS — All Four Configurations")
    print("-" * 70)
    print(f"{'Configuration':<40} {'Exact Match':>12} {'CodeBLEU':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['label']:<40} {r['exact_match']:>11.4f}  {r['codebleu']:>9.4f}")
    print("-" * 70)

    print("\nCodeBLEU breakdown:")
    print(f"{'Configuration':<40} {'n-gram':>8} {'syntax':>8} {'dataflow':>10}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['label']:<40} {r['ngram_match']:>8.4f} "
            f"{r['syntax_match']:>8.4f} {r['dataflow_match']:>10.4f}"
        )

    with open("./eval_results/results_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to ./eval_results/results_summary.json")
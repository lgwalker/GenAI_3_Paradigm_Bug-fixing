from datasets import load_dataset

def get_pretrain_data(num_samples=50000):
    """Loads and shuffles the CodeSearchNet Java dataset."""
    ds = load_dataset("code_search_net", "java", split="train")
    return ds.shuffle(seed=42).select(range(num_samples))

def get_finetune_data():
    """Loads the CodeXGlue Code Refinement (medium) dataset."""
    return load_dataset("google/code_x_glue_cc_code_refinement", name="medium")

def preprocess_finetune(examples, tokenizer):
    """Tokenizes buggy inputs and fixed targets for fine-tuning."""
    inputs = ["fix: " + b for b in examples["buggy"]]
    model_inputs = tokenizer(
        inputs,
        text_target=examples["fixed"], 
        max_length=512,
        truncation=True,
        padding=False
    )
    return model_inputs

import os
import torch
from tokenizer_utils import train_java_tokenizer
from dataset_utils import get_pretrain_data
from models import init_model_from_scratch
from collators import ManualSpanCorruptionCollator
from transformers import Trainer, TrainingArguments

def run():
    # Setup directories
    print("Setting up directories...")
    dirs = ["./java_tokenizer", "./final_pretrained_model", "./t5_pretrain"]
    for d in dirs: os.makedirs(d, exist_ok=True)

    # Data & Tokenizer
    print("Loading CodeSearchNet Java dataset...")
    ds = get_pretrain_data()
    corpus_path = "java_methods.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for code in ds["whole_func_string"]:
            f.write(code + "\n")
    print(f"Corpus written to {corpus_path}.")

    print("Training Java tokenizer...")
    tokenizer = train_java_tokenizer(corpus_path)
    print("Java tokenizer trained.")

    # Model
    model = init_model_from_scratch(len(tokenizer))
    
    # Tokenize Pre-training Set
    def tokenize_fn(x):
        return tokenizer(x["whole_func_string"], truncation=True, max_length=512)
    
    tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    tokenized_ds = tokenized_ds.filter(lambda x: 10 <= len(x["input_ids"]) <= 512)

    # Training
    args = TrainingArguments(
        output_dir="./t5_pretrain",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
        logging_steps=500,
        save_strategy="no",
        logging_strategy="epoch",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        warmup_steps=500,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=ManualSpanCorruptionCollator(tokenizer)
    )

    trainer.train()
    model.save_pretrained("./final_pretrained_model")
    tokenizer.save_pretrained("./final_pretrained_model")

if __name__ == "__main__":
    run()

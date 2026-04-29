import torch
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from dataset_utils import get_finetune_data, preprocess_finetune
from tokenizer_utils import load_tokenizer
from models import init_model_from_scratch

def train_pipeline(model, tokenizer, dataset, output_dir):
    """Standard fine-tuning loop for bug-fixing task."""
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,            # Set it higher than you think you need
        greater_is_better=False,        # Lower loss is better
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        warmup_steps=200,
        weight_decay=0.01,
        eval_strategy="epoch",          # Evaluate after each epoch
        save_strategy="epoch",          # Save after each epoch
        save_total_limit=1,             # Keep only best checkpoint on disk
        load_best_model_at_end=True,    # Load best val-loss model at end
        metric_for_best_model="eval_loss",
        logging_steps=100,
        logging_strategy="steps",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel and tokenizer saved to {output_dir}")
    return trainer

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    ds = get_finetune_data()
    tokenized_ds = ds.map(lambda x: preprocess_finetune(x, tokenizer), batched=True)

    # Pipeline A With Pre-training
    print("\n" + "-"*60)
    print("PIPELINE A — With Pre-training")
    print("-"*60)
    model_a = T5ForConditionalGeneration.from_pretrained("./final_pretrained_model")
    trainer_a = train_pipeline(model_a, tokenizer, tokenized_ds, "./pipeline_a_finetuned")
    print("\nPipeline A — Validation loss per epoch:")
    for entry in trainer_a.state.log_history:
        if "eval_loss" in entry:
            print(f"  Epoch {entry['epoch']:.0f}  |  eval_loss: {entry['eval_loss']:.4f}")
    del model_a
    torch.cuda.empty_cache()

    # Pipeline B Without Pre-training
    print("\n" + "-"*60)
    print("PIPELINE B — Without Pre-training (Initialized from Scratch)")
    print("-"*60)
    model_b = init_model_from_scratch(len(tokenizer))
    trainer_b = train_pipeline(model_b, tokenizer, tokenized_ds, "./pipeline_b_finetuned")
    print("\nPipeline B — Validation loss per epoch:")
    for entry in trainer_b.state.log_history:
        if "eval_loss" in entry:
            print(f"  Epoch {entry['epoch']:.0f}  |  eval_loss: {entry['eval_loss']:.4f}")
    del model_b
    torch.cuda.empty_cache()
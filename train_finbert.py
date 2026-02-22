"""
Project: GPU-Accelerated Sentiment Analysis on Local Workstation
Author: Jad Chamseddine
Description: Fine-tunes FinBERT on the Financial PhraseBank dataset,
optimized for an NVIDIA RTX 3060 (12GB VRAM).
"""
#==========================
# 1. Imports & dependencies
#==========================
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

#=========================
# 2. Hardware verification
#=========================
print("---Hardware verification---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Target device: {device}")

if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("--------------------------------\n")

#================================
# 3. Data loading & preprocessing
#================================
model_name = "ProsusAI/finbert"
dataset_name = "gtfintechlab/financial_phrasebank_sentences_allagree"

print("Loading dataset and tokenizer...")
# Dataset is already the sentences_allagree subset (100% annotator agreement — cleaner labels)
dataset = load_dataset(dataset_name, "5768")

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # max_length=128 keeps VRAM under control on the RTX 3060
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

#==================================
# 4. Model initialization & metrics
#==================================
print("Loading FinBERT model...")
# 3 labels: Positive (0), Negative (1), Neutral (2)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, use_safetensors=True)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

#===================
# 5. Training config
#===================
training_args = TrainingArguments(
    output_dir="./finbert_results",
    eval_strategy="epoch",
    learning_rate=2e-5,           # Standard fine-tuning LR for BERT-based models

    # Training hyperparameters
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    fp16=True,                       # Mixed precision — halves VRAM usage and speeds up on Tensor Cores

    num_train_epochs=3,
    weight_decay=0.01,            # L2 regularization to prevent overfitting
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

#=============
# 6. Execution
#=============
print("Starting local GPU fine-tuning...")
trainer.train()

print("Training complete! Saving model...")
trainer.save_model("./finbert_quant_finetuned")
tokenizer.save_pretrained("./finbert_quant_finetuned")
print("Model saved in './finbert_quant_finetuned'.")

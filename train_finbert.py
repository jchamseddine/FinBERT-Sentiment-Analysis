"""
Project: GPU-Accelerated Sentiment Analysis on Local Workstation
Author: Jad Chamseddine
Description : 
This script fine-tunes the FinBERT LLM on the Financial PhraseBank dataset.
It is optimized for local execution on an NVIDIA RTX 3060 (12GB VRAM) using advanced 
memory management techniques like Mixed Precision Training (fp16) and Gradient Accumulation.
"""
#==========================
# 1. Imports & dependencies
#==========================
# PyTorch: The core Deep Learning framework handling tensor operations on the GPU
import torch

import numpy as np

# Hugging Face Datasets: An efficient library to download, cache, and preprocess large datasets
from datasets import load_datasets
for transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Hugging Face Evaluate: Used to load standard evaluation metrics (like accuracy)
import evaluate

#=========================
# 2. Hardware verification
#=========================
print("---Hardware verification---")
#Check if a CUDA GPU is available if not fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Target device: {device}")

if device == "cuda":
    # Prints the name of the GPU
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("--------------------------------\n")

#================================
# 3. Data loading & preprocessing
#================================
model_name = "ProsusAI/finbert"
dataset_name = "financial_phrasebank"

print("Loading dataset and tokenizer...")
# We use the 'sentences_allagree' subset where 100% of human annotators agreed on the sentiment
# Relying on absolute human consensus eliminates noisy data and ensures a robust baseline
dataset = load_dataset(dataset_name, "sentences_allagree")

# The dataset only comes with a 'train' split. We manually create a 80/20 train/test split
dataset = dataset["train"].train_test_split(test_size=0.2, seed=69)

# Load automatically the tokenizer corresponding to the FinBERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    """
    Converts raw text sentences into numerical token IDs for the model.
    
    This function applies padding and truncation to ensure all tensors have
    the exact same size. This uniform shape is mandatory for efficient GPU 
    matrix operations and crucial to prevent Out-Of-Memory (OOM) errors on 
    a 12GB VRAM GPU.
    
    Args:
        examples (dict): A batch of data from the Hugging Face dataset, 
                         containing the key "sentence" with a list of raw text strings.
                         
    Returns:
        dict: A dictionary containing the tokenized outputs, specifically:
              - 'input_ids': The numerical sequence representing the text.
              - 'attention_mask': A binary mask indicating which tokens are actual words (1) and which are padding (0).
    """
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

#==================================
# 4. Model initialization & metrics
#==================================
print("Loading FinBERT model...")
# Initialize the model with 3 output labels: Positive (0), Negative (1), Neutral (2)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Load the accuracy metric wrapper.
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """
    Computes accuracy for evaluating model performance.

    This function is automatically called by the Hugging Face 'Trainer' 
    at the end of each evaluation step.

    Args:
        eval_pred (tuple): A named tuple containing two elements provided by the model:
            1. logits (np.ndarray): The raw, unnormalized scores predicted by the 
               model for each class (Positive, Negative, Neutral). 
               Shape: (batch_size, 3).
            2. labels (np.ndarray): The ground truth labels (actual categories).
               Shape: (batch_size,).

    Returns:
        dict: A dictionary containing the metric name and its calculated value.
              Format: {"accuracy": float_value}
              Example: {"accuracy": 0.92}
    """
    # 1. Unpack the raw predictions (logits) and the true results (labels)
    logits, labels = eval_pred
    
    # 2. Transform raw scores into a single prediction:
    # np.argmax picks the index of the highest value (e.g., [0.1, 0.8, 0.1] -> 1)
    predictions = np.argmax(logits, axis=-1)
    
    # 3. Pass predictions and true labels to the evaluation tool
    # The .compute() method compares both lists and returns the success percentage
    return accuracy_metric.compute(predictions=predictions, references=labels)

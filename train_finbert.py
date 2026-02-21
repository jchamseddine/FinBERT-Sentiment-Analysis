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
# PyTorch: The core Deep Learning framework handling tensor operations on the GPU.
import torch

import numpy as np

# Hugging Face Datasets: An efficient library to download, cache, and preprocess large datasets.
from datasets import load_datasets
for transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Hugging Face Evaluate: Used to load standard evaluation metrics (like accuracy).
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

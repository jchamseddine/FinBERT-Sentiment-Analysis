# FinBERT Sentiment Analysis

Fine-tuning FinBERT on the Financial PhraseBank dataset for financial sentiment analysis, optimized for local GPU execution.

## Hardware
- NVIDIA RTX 3060 (12GB VRAM)
- Mixed Precision Training (fp16)
- Gradient Accumulation

## Model
- Base model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Dataset: [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank)

## Setup

```bash
pip install -r requirements.txt
python train_finbert.py
```

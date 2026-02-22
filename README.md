# FinBERT Sentiment Analysis

Fine-tuning FinBERT on the Financial PhraseBank dataset for financial sentiment analysis, optimized for local GPU execution.

## Hardware
- NVIDIA RTX 3060 (12GB VRAM)
- Mixed Precision Training (fp16)

## Model & Data
- Base model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Dataset: [gtfintechlab/financial_phrasebank_sentences_allagree](https://huggingface.co/datasets/gtfintechlab/financial_phrasebank_sentences_allagree) — subset with 100% annotator agreement (config `5768`)
- Labels: Positive (0), Negative (1), Neutral (2)

## Output
The fine-tuned model and tokenizer are saved to `./finbert_quant_finetuned/` after training.

## Setup

```bash
conda create -n finbert python=3.10
conda activate finbert
# Install PyTorch with CUDA 12.1 support (required for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python train_finbert.py
```

## Dependencies
- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `numpy`
- `accelerate`

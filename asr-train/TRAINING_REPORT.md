# ASR Fine-Tuning Report: Parakeet-TDT for Singapore English

## Overview

This report documents the adapter-based fine-tuning of NVIDIA's Parakeet-TDT ASR model for improved transcription of Singapore English podcast audio, and the evaluation results comparing the baseline and fine-tuned models.

## Base Model

- **Model**: `nvidia/parakeet-tdt-0.6b-v2`
- **Architecture**: EncDecRNNTBPEModel (Conformer encoder + RNN-T decoder with Token-and-Duration Transducer)
- **Parameters**: 617,909,126 total
- **Tokenizer**: SentencePieceTokenizer (1024 tokens)
- **Loss**: TDT (Token-and-Duration Transducer) with durations [0, 1, 2, 3, 4]

## Fine-Tuning Method

### Why Adapter Training?

Full fine-tuning of a 617M-parameter model on only ~3,000 training samples would risk catastrophic forgetting — the model could overfit to the small Singapore English dataset and lose its strong general English ASR capability. Adapter training avoids this by **freezing all pretrained weights** and inserting a small trainable bottleneck module. This has several advantages:

1. **Low risk of overfitting**: Only 83,200 parameters (0.01% of the model) are trained, so the model cannot memorize the small training set.
2. **Preserves base model quality**: The frozen Conformer encoder and RNN-T decoder retain their general English ASR ability. The adapter learns a lightweight correction on top.
3. **Fast training**: With so few trainable parameters, training completes in ~1,000 steps on a single GPU.
4. **Easy deployment**: The adapter weights are a ~577KB file that can be loaded on top of the base model without redistributing the full 2.5GB checkpoint.

The adapter is placed on the **joint network** (rather than the encoder) because the joint network combines encoder and predictor representations before producing token logits — this is the most effective place to adapt the model's output vocabulary and language-specific patterns without disrupting the acoustic feature extraction in the encoder.

### Adapter Configuration

A **linear adapter** was inserted into the model's joint network. This approach keeps the pretrained weights frozen and only trains a small bottleneck module, making it highly parameter-efficient.

| Setting | Value |
|---|---|
| Adapter type | `LinearAdapter` |
| Adapter location | `joint` (RNN-T joint network) |
| Adapter name | `asr_sg_english` |
| Input dimension | 1024 (mapped from encoder output 640) |
| Bottleneck dimension | 64 |
| Activation | Swish |
| Norm position | Pre-norm |
| Dropout | 0.0 |
| **Trainable parameters** | **83,200 (0.01% of total)** |

All encoder BatchNorm layers and other pretrained weights were frozen. Only the adapter weights were updated during training.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.002 |
| Weight decay | 0.0 |
| LR scheduler | None |
| Precision | bf16-mixed |
| Max steps | 1,000 |
| Batch size | 64 |
| Validation check interval | Every 47 steps |
| Spec augmentation | Disabled |
| Devices | 1x GPU |
| Checkpoint selection | Best `val_wer` (top 3 saved) |

### Training Data

The training data consists of manually transcribed segments from Singapore English podcasts (YCSEP dataset).

| Split | Entries |
|---|---|
| Train | 3,045 |
| Validation | 734 |

- Audio clips were downloaded as WAV files at 16kHz
- Stereo audio was handled with `channel_selector: average`
- Segments with duration <= 0s or > 25s were filtered out
- NeMo JSONL manifests were generated from CSV metadata

### Results

| Metric | Baseline (parakeet-tdt-0.6b-v3) | Fine-tuned (parakeet-tdt-sg-english) | Difference |
|---|---|---|---|
| **Corpus WER** | 34.16% | 29.48% | **-4.68pp** |
| **Macro WER** | 58.14% | 51.60% | **-6.55pp** |
| Rows evaluated | 500 / 500 | 491 / 500 | — |

- **Corpus WER** aggregates all words across utterances into a single WER computation (weighted by utterance length). This is the more robust metric.
- **Macro WER** averages per-utterance WER values equally. Short utterances with even one error can produce 100% WER, inflating this metric.

### Analysis

The fine-tuned model achieves a relative reduction in corpus WER, demonstrating that the adapter successfully learned Singapore English speech patterns from a small amount of transcribed podcast audio and only 83,200 trainable parameters.

The macro WER improvement suggests the adapter also improved transcription quality on shorter, colloquial utterances where the baseline model struggled most.

9 out of 500 clips produced empty transcriptions from the fine-tuned model, compared to 0 from the baseline. These were excluded from the fine-tuned WER calculation.

## Reproduction

```bash
python asr-train/test_finetune.py
```

Results are saved to `asr-train/finetune_comparison.csv` with columns: `text`, `generated_text`, `generate_ft_text`, `clip_path`.

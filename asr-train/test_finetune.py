import pandas as pd
import numpy as np
import librosa
import torch
from jiwer import wer
from pathlib import Path
from tqdm import tqdm

import nemo.collections.asr as nemo_asr

SAMPLE_SIZE = 500
RANDOM_SEED = 42
BATCH_SIZE = 16

CSV_PATH = Path(__file__).resolve().parent.parent / "asr" / "data" / "TDK_subset.csv"
FT_MODEL = "joentze/parakeet-tdt-sg-english"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model = model.to(device)
    model.eval()
    return model


def load_audio(path: str, sr: int = 16_000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def transcribe_batch(model, audio_paths: list[str], batch_size: int = BATCH_SIZE) -> list[str]:
    all_transcriptions = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Transcribing"):
        batch_paths = audio_paths[i : i + batch_size]
        batch = [load_audio(p) for p in batch_paths]
        with torch.no_grad():
            outputs = model.transcribe(batch)
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], str):
            all_transcriptions.extend(outputs)
        else:
            all_transcriptions.extend([o.text if hasattr(o, "text") else str(o) for o in outputs])
    return all_transcriptions


def compute_wer(references: list[str], hypotheses: list[str]) -> dict:
    refs_clean = [r.strip() for r in references]
    hyps_clean = [h.strip() if isinstance(h, str) else "" for h in hypotheses]

    valid = [(r, h) for r, h in zip(refs_clean, hyps_clean) if r and h]
    refs_valid, hyps_valid = zip(*valid) if valid else ([], [])

    corpus_wer = wer(list(refs_valid), list(hyps_valid))

    per_utt = [wer(r, h) for r, h in zip(refs_valid, hyps_valid)]
    macro_wer = sum(per_utt) / len(per_utt) if per_utt else 0.0

    return {
        "corpus_wer": corpus_wer,
        "macro_wer": macro_wer,
        "n_evaluated": len(refs_valid),
        "n_skipped": len(references) - len(refs_valid),
    }


def main():
    print(f"Loading CSV from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    df = df.dropna(subset=["text", "generated_text", "clip_path"])
    df = df[df["text"].str.strip().astype(bool)]
    df = df[df["clip_path"].apply(lambda p: Path(p).exists())]

    print(f"Valid rows with existing audio: {len(df)}")

    if len(df) < SAMPLE_SIZE:
        print(f"Warning: only {len(df)} rows available (requested {SAMPLE_SIZE}). Using all.")
        sample = df.copy()
    else:
        sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).copy()

    print(f"\nSampled {len(sample)} rows\n")

    # --- Baseline WER (existing generated_text vs text) ---
    print("=" * 60)
    print("BASELINE: generated_text vs text")
    print("=" * 60)
    baseline = compute_wer(sample["text"].tolist(), sample["generated_text"].tolist())
    print(f"  Corpus WER : {baseline['corpus_wer']:.4f} ({baseline['corpus_wer'] * 100:.2f}%)")
    print(f"  Macro  WER : {baseline['macro_wer']:.4f} ({baseline['macro_wer'] * 100:.2f}%)")
    print(f"  Evaluated  : {baseline['n_evaluated']} / {len(sample)}")

    # --- Fine-tuned model transcription ---
    print(f"\nLoading fine-tuned model: {FT_MODEL}")
    device = pick_device()
    print(f"Using device: {device}")
    model = load_model(FT_MODEL, device)

    audio_paths = sample["clip_path"].tolist()
    print(f"Transcribing {len(audio_paths)} clips...")
    ft_transcriptions = transcribe_batch(model, audio_paths)
    sample["generate_ft_text"] = ft_transcriptions

    # --- Fine-tuned WER ---
    print("\n" + "=" * 60)
    print("FINE-TUNED: generate_ft_text vs text")
    print("=" * 60)
    ft_result = compute_wer(sample["text"].tolist(), sample["generate_ft_text"].tolist())
    print(f"  Corpus WER : {ft_result['corpus_wer']:.4f} ({ft_result['corpus_wer'] * 100:.2f}%)")
    print(f"  Macro  WER : {ft_result['macro_wer']:.4f} ({ft_result['macro_wer'] * 100:.2f}%)")
    print(f"  Evaluated  : {ft_result['n_evaluated']} / {len(sample)}")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    corpus_diff = ft_result["corpus_wer"] - baseline["corpus_wer"]
    macro_diff = ft_result["macro_wer"] - baseline["macro_wer"]
    print(f"  {'Metric':<20} {'Baseline':>12} {'Fine-tuned':>12} {'Diff':>12}")
    print(f"  {'-' * 56}")
    print(
        f"  {'Corpus WER':<20} {baseline['corpus_wer']:>11.2%} {ft_result['corpus_wer']:>11.2%} {corpus_diff:>+11.2%}"
    )
    print(
        f"  {'Macro WER':<20} {baseline['macro_wer']:>11.2%} {ft_result['macro_wer']:>11.2%} {macro_diff:>+11.2%}"
    )

    if corpus_diff < 0:
        print(f"\n  Fine-tuned model is BETTER by {abs(corpus_diff) * 100:.2f}pp (corpus WER)")
    elif corpus_diff > 0:
        print(f"\n  Fine-tuned model is WORSE by {corpus_diff * 100:.2f}pp (corpus WER)")
    else:
        print("\n  Models perform identically (corpus WER)")

    # Save results
    out_path = Path(__file__).resolve().parent / "finetune_comparison.csv"
    sample[["text", "generated_text", "generate_ft_text", "clip_path"]].to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

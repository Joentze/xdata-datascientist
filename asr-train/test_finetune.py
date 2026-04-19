
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import numpy as np
import pandas as pd
import requests
import torch
from jiwer import wer
from tqdm import tqdm

SAMPLE_SIZE = 500
RANDOM_SEED = 42
BATCH_SIZE = 16

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "asr" / "TDK_subset.csv"
CLIP_DIR = REPO_ROOT / "asr" / "data" / "clips"

FT_MODEL = "joentze/parakeet-tdt-sg-english"

CLIP_API = "https://ycsep-fastapi-viewer-ycsep-test.2.rahtiapp.fi/clip"
AUDIO_BASE = "https://a3s.fi/swift/v1/YCSEP_v2"


def parse_viewer_url(viewer_url: str) -> str:
    segment_id = unquote(urlparse(viewer_url).path.rstrip("/").split("/")[-1])
    body = re.sub(r"_\d+_\d+$", "", segment_id)
    match = re.search(r"(\d{8}--)", body)
    if not match:
        raise ValueError(f"Cannot parse viewer URL: {viewer_url}")
    filename_slug = body[match.start() :]
    date, video_id, title_slug = filename_slug.split("--", 2)
    title = title_slug.replace("-", " ")
    return f"{AUDIO_BASE}/{date}--{video_id}--{quote(title)}.wav"


def download_clip(
    audio_url: str,
    start: float,
    end: float,
    output_dir: Path,
    timeout: int = 60,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(unquote(urlparse(audio_url).path).split("/")[-1]).stem
    filename = f"{stem}_{int(start * 1000)}_{int(end * 1000)}.wav"
    dest = output_dir / filename

    if dest.exists():
        return dest

    resp = requests.get(
        CLIP_API,
        params={"url": audio_url, "start": start, "end": end, "fmt": "wav"},
        timeout=timeout,
    )
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def download_clips_for_sample(df: pd.DataFrame) -> pd.DataFrame:
    clip_paths = []
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading clips"):
        try:
            audio_url = parse_viewer_url(row["audio"])
            clip_path = download_clip(
                audio_url, row["start_time"], row["end_time"], CLIP_DIR
            )
            clip_paths.append(str(clip_path))
        except Exception as e:
            print(f"  Failed row S={row['S']}: {e}")
            clip_paths.append("")
            failed += 1

    df = df.copy()
    df["clip_path_local"] = clip_paths
    print(f"Downloaded: {len(df) - failed}/{len(df)} clips ({failed} failed)")
    return df


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model = model.to(device)
    model.eval()
    return model


def load_audio(path: str, sr: int = 16_000) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def transcribe_batch(
    model, audio_paths: list[str], batch_size: int = BATCH_SIZE
) -> list[str]:
    all_transcriptions = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Transcribing"):
        batch_paths = audio_paths[i : i + batch_size]
        batch = [load_audio(p) for p in batch_paths]
        with torch.no_grad():
            outputs = model.transcribe(batch)
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], str):
            all_transcriptions.extend(outputs)
        else:
            all_transcriptions.extend(
                [o.text if hasattr(o, "text") else str(o) for o in outputs]
            )
    return all_transcriptions


def compute_wer(references: list[str], hypotheses: list[str]) -> dict:
    refs_clean = [r.strip() for r in references]
    hyps_clean = [h.strip() if isinstance(h, str) else "" for h in hypotheses]

    valid = [(r, h) for r, h in zip(refs_clean, hyps_clean) if r and h]
    if not valid:
        return {
            "corpus_wer": 0.0,
            "macro_wer": 0.0,
            "n_evaluated": 0,
            "n_skipped": len(references),
        }
    refs_valid, hyps_valid = zip(*valid)

    corpus_wer = wer(list(refs_valid), list(hyps_valid))
    per_utt = [wer(r, h) for r, h in zip(refs_valid, hyps_valid)]
    macro_wer = sum(per_utt) / len(per_utt) if per_utt else 0.0

    return {
        "corpus_wer": corpus_wer,
        "macro_wer": macro_wer,
        "n_evaluated": len(refs_valid),
        "n_skipped": len(references) - len(refs_valid),
    }


def print_wer(label: str, result: dict, total: int):
    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"  Corpus WER : {result['corpus_wer']:.4f} ({result['corpus_wer'] * 100:.2f}%)")
    print(f"  Macro  WER : {result['macro_wer']:.4f} ({result['macro_wer'] * 100:.2f}%)")
    print(f"  Evaluated  : {result['n_evaluated']} / {total}")


def main():
    print(f"Loading CSV from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df)}")

    df = df.dropna(subset=["text", "generated_text", "audio"])
    df = df[df["text"].str.strip().astype(bool)]
    print(f"Rows with valid text + generated_text + audio: {len(df)}")

    if len(df) == 0:
        print("Error: no valid rows found.")
        return

    sample_n = min(SAMPLE_SIZE, len(df))
    sample = df.sample(n=sample_n, random_state=RANDOM_SEED).copy()
    print(f"Sampled {len(sample)} rows\n")

    # --- Download clips ---
    print("Downloading audio clips...")
    sample = download_clips_for_sample(sample)
    sample = sample[sample["clip_path_local"] != ""]
    print(f"Rows with successfully downloaded clips: {len(sample)}\n")

    if len(sample) == 0:
        print("Error: no clips were downloaded successfully.")
        return

    # --- Baseline WER (existing generated_text vs text) ---
    baseline = compute_wer(sample["text"].tolist(), sample["generated_text"].tolist())
    print_wer("BASELINE: generated_text vs text", baseline, len(sample))

    # --- Fine-tuned model transcription ---
    print(f"\nLoading fine-tuned model: {FT_MODEL}")
    device = pick_device()
    print(f"Using device: {device}")
    model = load_model(FT_MODEL, device)

    audio_paths = sample["clip_path_local"].tolist()
    print(f"Transcribing {len(audio_paths)} clips...")
    ft_transcriptions = transcribe_batch(model, audio_paths)
    sample["generate_ft_text"] = ft_transcriptions

    # --- Fine-tuned WER ---
    ft_result = compute_wer(sample["text"].tolist(), sample["generate_ft_text"].tolist())
    print_wer("\nFINE-TUNED: generate_ft_text vs text", ft_result, len(sample))

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

    out_path = Path(__file__).resolve().parent / "finetune_comparison.csv"
    sample[["text", "generated_text", "generate_ft_text", "clip_path_local"]].to_csv(
        out_path, index=False
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

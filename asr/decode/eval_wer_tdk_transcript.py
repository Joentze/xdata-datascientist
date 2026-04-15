import pandas as pd
from jiwer import wer
from pathlib import Path

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "TDK_subset.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    required = {"text", "generated_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.dropna(subset=["text", "generated_text"])
    df = df[df["text"].str.strip().astype(
        bool) & df["generated_text"].str.strip().astype(bool)]

    if len(df) < SAMPLE_SIZE:
        print(
            f"Warning: only {len(df)} valid rows available (requested {SAMPLE_SIZE}). Using all.")
        sample = df
    else:
        sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    print(f"Evaluating WER on {len(sample)} rows...")

    references = sample["text"].tolist()
    hypotheses = sample["generated_text"].tolist()

    average_wer = wer(references, hypotheses)
    print(f"Average WER: {average_wer:.4f} ({average_wer * 100:.2f}%)")

    per_row_wers = [wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    macro_avg_wer = sum(per_row_wers) / len(per_row_wers)
    print(
        f"Macro-average WER (per-utterance): {macro_avg_wer:.4f} ({macro_avg_wer * 100:.2f}%)")


if __name__ == "__main__":
    main()

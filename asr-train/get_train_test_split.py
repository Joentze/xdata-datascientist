import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 4000
VAL_SIZE = 1000
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
SRC_CSV = BASE_DIR.parent / "asr" / "data" / "non_TDK_subset.csv"
OUT_DIR = BASE_DIR / "data"


def main():
    df = pd.read_csv(SRC_CSV)
    total_needed = TRAIN_SIZE + VAL_SIZE

    print(f"Loaded {len(df)} rows from {SRC_CSV.name}")
    if len(df) < total_needed:
        raise ValueError(
            f"Not enough rows: need {total_needed}, got {len(df)}"
        )

    subset = df.sample(n=total_needed, random_state=RANDOM_SEED)

    train_df, val_df = train_test_split(
        subset,
        train_size=TRAIN_SIZE,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUT_DIR / "train.csv"
    val_path = OUT_DIR / "validation.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train set: {len(train_df)} rows -> {train_path}")
    print(f"Validation set: {len(val_df)} rows -> {val_path}")


if __name__ == "__main__":
    main()

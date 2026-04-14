import csv
from pathlib import Path

from pydub import AudioSegment

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "TDK_subset.csv"
AUDIO_DIR = BASE_DIR / "audio"
CLIP_DIR = BASE_DIR / "clip"
CLIP_COLUMN = "clip_path"


def count_data_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        next(reader, None)
        return sum(1 for _ in reader)


def to_ms(seconds: str) -> int:
    return max(0, int(float(seconds) * 1000))


def source_audio_path(file_field: str) -> Path:
    textgrid_name = Path(file_field).name
    audio_name = f"{Path(textgrid_name).stem}.mp3"
    return AUDIO_DIR / audio_name


def clip_name(row: dict[str, str], start_ms: int, end_ms: int) -> str:
    source_stem = Path(row["file"]).stem
    row_id = row.get("S", "row")
    return f"{source_stem}__{row_id}_{start_ms}_{end_ms}.mp3"


def main() -> None:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = CSV_PATH.with_suffix(".tmp.csv")
    total_rows = count_data_rows(CSV_PATH)

    current_audio_path: Path | None = None
    current_audio_segment: AudioSegment | None = None
    processed = 0
    missing_audio = 0

    with CSV_PATH.open("r", newline="", encoding="utf-8") as infile, temp_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        if CLIP_COLUMN not in fieldnames:
            fieldnames.append(CLIP_COLUMN)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, row in enumerate(reader, start=1):
            print(f"Processing row {row_index}/{total_rows}", flush=True)
            start_ms = to_ms(row["start_time"])
            end_ms = max(start_ms, to_ms(row["end_time"]))
            full_audio_path = source_audio_path(row["file"])

            if not full_audio_path.exists():
                row[CLIP_COLUMN] = ""
                missing_audio += 1
                writer.writerow(row)
                continue

            if current_audio_path != full_audio_path:
                current_audio_segment = AudioSegment.from_file(full_audio_path)
                current_audio_path = full_audio_path

            clip_path = CLIP_DIR / clip_name(row, start_ms, end_ms)
            if not clip_path.exists():
                clip_segment = current_audio_segment[start_ms:end_ms]
                clip_segment.export(clip_path, format="mp3")

            row[CLIP_COLUMN] = str(clip_path)
            writer.writerow(row)
            processed += 1

    temp_path.replace(CSV_PATH)
    print(f"Done. Processed rows: {processed}")
    print(f"Rows with missing source audio: {missing_audio}")
    print(f"Updated CSV: {CSV_PATH}")
    print(f"Clips written to: {CLIP_DIR}")


if __name__ == "__main__":
    main()

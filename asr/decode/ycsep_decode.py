import asyncio
import csv
import sys
from pathlib import Path

import httpx

API_ENDPOINT = "http://localhost:8001"
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "TDK_subset.csv"
TMP_CSV_PATH = CSV_PATH.with_name(f"{CSV_PATH.stem}.tmp{CSV_PATH.suffix}")
BATCH_SIZE = 50
ASR_TIMEOUT = 120

asr_client = httpx.AsyncClient(base_url=API_ENDPOINT, timeout=ASR_TIMEOUT)


async def ping():
    response = await asr_client.get("/ping")
    data = response.json()
    if data.get("message") != "pong":
        print("API ping failed, exiting.")
        sys.exit(1)
    print("API is up.")


async def process_row(row: dict) -> dict:
    clip_path_value = (row.get("clip_path") or "").strip()
    if not clip_path_value:
        raise ValueError("Missing clip_path")

    clip_path = Path(clip_path_value)
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip does not exist: {clip_path}")

    with clip_path.open("rb") as f:
        files = {"file": (clip_path.name, f, "audio/mpeg")}
        response = await asr_client.post("/asr", files=files)
    response.raise_for_status()
    data = response.json()
    return {
        "generated_text": data["transcription"],
        "duration": data["duration"],
    }


def ensure_output_columns(fieldnames: list[str]) -> list[str]:
    updated = list(fieldnames)
    for name in ("generated_text", "duration"):
        if name not in updated:
            updated.append(name)
    return updated


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def main():
    try:
        await ping()

        read_path = TMP_CSV_PATH if TMP_CSV_PATH.exists() else CSV_PATH
        if read_path == TMP_CSV_PATH:
            print(f"Resuming from checkpoint: {TMP_CSV_PATH}")

        with read_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = ensure_output_columns(list(reader.fieldnames or []))
            rows = list(reader)

        total = len(rows)
        print(
            f"Starting transcription of {total} files (batch_size={BATCH_SIZE})")

        result_map: dict[int, dict] = {}
        for i in range(0, total, BATCH_SIZE):
            batch = rows[i: i + BATCH_SIZE]
            batch_indices = list(range(i, i + len(batch)))
            pending = [
                (idx, row) for idx, row in zip(batch_indices, batch)
                if not (row.get("generated_text") or "").strip()
                and not (row.get("duration") or "").strip()
            ]
            if not pending:
                print(f"Skipping batch {i + 1}-{i + len(batch)} (already done)")
                continue

            tasks = [process_row(row) for _, row in pending]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (idx, _), result in zip(pending, results):
                if isinstance(result, Exception):
                    print(f"FAILED row {idx + 1}/{total}: {result}")
                    continue
                result_map[idx] = result
                truncated = result["generated_text"]
                if len(truncated) > 50:
                    truncated = truncated[:50] + "..."
                print(
                    f"{len(result_map)}/{total} done | row {idx + 1}/{total} | {result['duration']}s | {truncated}")

            for idx in [k for k in result_map if i <= k < i + len(batch)]:
                rows[idx]["generated_text"] = result_map[idx]["generated_text"]
                rows[idx]["duration"] = result_map[idx]["duration"]

            write_csv(TMP_CSV_PATH, rows, fieldnames)
            print(f"Checkpoint saved: {TMP_CSV_PATH}")

        write_csv(CSV_PATH, rows, fieldnames)
        TMP_CSV_PATH.unlink(missing_ok=True)

        print(f"Done. Wrote generated_text and duration columns to {CSV_PATH}")
    finally:
        await asr_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())

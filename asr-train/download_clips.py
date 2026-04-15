import asyncio
import csv
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr.decode.download_audio import CLIP_API, clip_filename, parse_viewer_url

BASE_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 20
TIMEOUT = 60

SPLITS = {
    "train": {
        "csv": BASE_DIR / "data" / "train.csv",
        "out": BASE_DIR / "data" / "train",
    },
    "validate": {
        "csv": BASE_DIR / "data" / "validation.csv",
        "out": BASE_DIR / "data" / "validate",
    },
}


async def download_one(
    client: httpx.AsyncClient,
    url: str,
    start: float,
    end: float,
    output_dir: Path,
) -> Path | None:
    dest = output_dir / clip_filename(url, start, end)
    if dest.exists():
        return dest

    try:
        resp = await client.get(
            CLIP_API,
            params={"url": url, "start": start, "end": end, "fmt": "wav"},
        )
        resp.raise_for_status()
        if len(resp.content) == 0:
            print(f"  EMPTY {dest.name} — skipping")
            return None
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        print(f"  FAILED {dest.name}: {e}")
        return None


async def download_split(name: str, csv_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"\n[{name}] {total} clips -> {output_dir}")

    done = 0
    skipped = 0
    failed = 0

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for i in range(0, total, BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            tasks = []
            for row in batch:
                url = parse_viewer_url(row["audio"])
                start = float(row["start_time"])
                end = float(row["end_time"])
                dest = output_dir / clip_filename(url, start, end)
                if dest.exists():
                    skipped += 1
                    continue
                tasks.append(download_one(client, url, start, end, output_dir))

            if tasks:
                results = await asyncio.gather(*tasks)
                for r in results:
                    if r is None:
                        failed += 1
                    else:
                        done += 1

            progress = skipped + done + failed
            print(
                f"  [{name}] {progress}/{total}  "
                f"(downloaded: {done}, skipped: {skipped}, failed: {failed})"
            )

    print(f"[{name}] Finished — {done} downloaded, {skipped} skipped, {failed} failed")


async def main():
    for name, cfg in SPLITS.items():
        await download_split(name, cfg["csv"], cfg["out"])


if __name__ == "__main__":
    asyncio.run(main())

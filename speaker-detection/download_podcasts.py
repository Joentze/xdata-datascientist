import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import requests

AUDIO_BASE = "https://a3s.fi/swift/v1/YCSEP_v2"
EPISODES_CSV = Path(__file__).resolve().parent / "data" / "unique_podcasts.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "podcasts"
PATHS_CSV = Path(__file__).resolve().parent / "data" / "tmp_podcast_paths.csv"
BATCH_SIZE = 50
MAX_WORKERS = 4
TIMEOUT = 120


def load_episodes(csv_path: Path) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["episode_url"].strip() for row in reader if row["episode_url"].strip()]


def load_completed(csv_path: Path) -> dict[str, str]:
    """Load already-downloaded episodes from a previous run's paths CSV."""
    completed = {}
    if not csv_path.exists():
        return completed
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row["episode_url"].strip()
            path = row["local_path"].strip()
            if url and path and Path(path).exists():
                completed[url] = path
    return completed


def viewer_url_to_wav(viewer_url: str) -> tuple[str, Path]:
    """Convert a viewer URL to (remote wav URL, local wav path)."""
    segment_id = unquote(urlparse(viewer_url).path.rstrip("/").split("/")[-1])
    body = re.sub(r"_\d+_\d+$", "", segment_id)
    match = re.search(r"(\d{8}--)", body)
    slug = body[match.start():]
    date, video_id, title_slug = slug.split("--", 2)
    title = title_slug.replace("-", " ")
    remote = f"{AUDIO_BASE}/{date}--{video_id}--{quote(title)}.wav"
    local = OUTPUT_DIR / f"{date}--{video_id}--{title}.wav"
    return remote, local


def download_one(episode_url: str) -> tuple[str, bool, str]:
    """Returns (episode_url, success, message)."""
    try:
        remote_url, dest = viewer_url_to_wav(episode_url)
        if dest.exists():
            print(f"Already exists, skipping: {dest.name}")
            return episode_url, True, str(dest)
        resp = requests.get(remote_url, timeout=TIMEOUT)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return episode_url, True, str(dest)
    except Exception as e:
        return episode_url, False, str(e)


def main():
    all_episodes = load_episodes(EPISODES_CSV)
    total = len(all_episodes)

    completed = load_completed(PATHS_CSV)
    skipped = len(completed)

    episodes = [ep for ep in all_episodes if ep not in completed]
    remaining = len(episodes)

    print(f"Loaded {total} episodes from {EPISODES_CSV}")
    if skipped:
        print(f"Resuming: {skipped} already downloaded, {remaining} remaining\n")
    else:
        print(f"{remaining} to download\n")

    if remaining == 0:
        print("Nothing to do — all episodes already downloaded.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done = skipped
    failed = 0
    failed_episodes = []

    with open(PATHS_CSV, "a", newline="", encoding="utf-8") as paths_file:
        writer = csv.writer(paths_file)
        if skipped == 0:
            writer.writerow(["episode_url", "local_path"])

        for batch_start in range(0, remaining, BATCH_SIZE):
            batch = episodes[batch_start: batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            batch_total = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
            print(
                f"--- Batch {batch_num}/{batch_total} ({len(batch)} episodes) ---")

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(download_one, ep): ep for ep in batch}
                for future in as_completed(futures):
                    url, ok, msg = future.result()
                    done += 1
                    if ok:
                        writer.writerow([url, msg])
                        paths_file.flush()
                        print(f"  [{done}/{total}] OK: {msg}")
                    else:
                        failed += 1
                        failed_episodes.append((url, msg))
                        print(f"  [{done}/{total}] FAIL: {msg}")

            elapsed = time.time() - t0
            print(f"  Batch done in {elapsed:.1f}s\n")

    print(f"Finished: {done - failed}/{total} succeeded, {failed} failed")
    print(f"Paths CSV: {PATHS_CSV}")
    if failed_episodes:
        print("\nFailed episodes:")
        for url, msg in failed_episodes:
            print(f"  {url}\n    -> {msg}")


if __name__ == "__main__":
    main()

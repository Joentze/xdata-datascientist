import json
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from pyannote.audio import Inference, Model
from scipy.spatial.distance import cdist

BASE_DIR = Path(
    "/Users/tanjoen/Documents/xdata-datascientist/speaker-detection")
SEGMENTS_JSON = BASE_DIR / "data" / "podcast_segments.json"
PODCASTS_DIR = BASE_DIR / "data" / "podcasts"
SPEAKER_REF = BASE_DIR / "speaker_ref.wav"
PROGRESS_FILE = BASE_DIR / "data" / "embed_progress.json"
MATCHES_FILE = BASE_DIR / "data" / "speaker_matches.json"
DOWNLOAD_TIMEOUT = 180
DOWNLOAD_WORKERS = 8
PREFETCH = 8
WINDOW_DURATION = 60.0
WINDOW_STEP = 30.0
SIMILARITY_THRESHOLD = 0.50


def podcast_key_to_local_path(key: str) -> Path:
    """Derive the local WAV path from a podcast key (mirrors download_podcasts.py)."""
    match = re.search(r"\d{8}--", key)
    slug = key[match.start():]
    date, video_id, title_slug = slug.split("--", 2)
    title = title_slug.replace("-", " ")
    return PODCASTS_DIR / f"{date}--{video_id}--{title}.wav"


def download_wav(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_progress() -> dict:
    """Return {podcast_key: result_dict_or_None} for already-processed keys."""
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        if isinstance(data, list):
            return {k: None for k in data}
        return data
    return {}


def save_progress(done: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(done, indent=2))


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: set the HF_TOKEN environment variable with your Hugging Face access token.")
        print("  1. Accept conditions at https://hf.co/pyannote/embedding")
        print("  2. Create a token at https://hf.co/settings/tokens")
        return

    if not SPEAKER_REF.exists():
        print(f"Reference WAV not found: {SPEAKER_REF}")
        return

    with open(SEGMENTS_JSON) as f:
        podcasts: dict = json.load(f)

    total_podcasts = len(podcasts)
    print(f"Loaded {total_podcasts} podcasts")

    done = load_progress()
    SKIP_SOURCES = {"Historyogi"}
    remaining = [
        (k, v) for k, v in podcasts.items()
        if k not in done and not any(s in k for s in SKIP_SOURCES)
    ]
    if done:
        print(
            f"Resuming: {len(done)} already processed, {len(remaining)} remaining")
    if not remaining:
        print("All podcasts already processed.")
        return

    # ── Model ──────────────────────────────────────────────────────────
    print("Loading pyannote/embedding model...")
    model = Model.from_pretrained(
        "pyannote/embedding", use_auth_token=hf_token)

    sliding_inference = Inference(
        model, window="sliding",
        duration=WINDOW_DURATION, step=WINDOW_STEP,
    )
    whole_inference = Inference(model, window="whole")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    sliding_inference.to(torch.device(device))
    whole_inference.to(torch.device(device))
    print(f"Inference device: {device}")

    # ── Embed speaker reference ────────────────────────────────────────
    print(f"Embedding speaker reference: {SPEAKER_REF.name}")
    ref_embedding = whole_inference(str(SPEAKER_REF))
    ref_embedding = np.array(ref_embedding).reshape(1, -1)
    print(f"Reference embedding shape: {ref_embedding.shape}")

    # ── Prefetch download pool ─────────────────────────────────────────
    PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
    dl_pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)

    dl_futures: dict[str, Future] = {}
    for key, podcast in remaining[:PREFETCH]:
        local = podcast_key_to_local_path(key)
        dl_futures[key] = dl_pool.submit(
            download_wav, podcast["wav_url"], local)

    # ── Main loop ──────────────────────────────────────────────────────
    processed = len(done)
    matches: list[dict] = [v for v in done.values() if v is not None]

    for i, (key, podcast) in enumerate(remaining):
        # Keep the prefetch window topped up
        lookahead = i + PREFETCH
        if lookahead < len(remaining):
            nk, np_ = remaining[lookahead]
            if nk not in dl_futures:
                dl_futures[nk] = dl_pool.submit(
                    download_wav, np_[
                        "wav_url"], podcast_key_to_local_path(nk),
                )

        wav_url = podcast["wav_url"]
        local_path = podcast_key_to_local_path(key)
        processed += 1

        print(f"\n[{processed}/{total_podcasts}] {key}")

        try:
            dl_futures.pop(key).result()
        except Exception as e:
            print(f"  SKIP (download failed): {e}")
            done[key] = None
            save_progress(done)
            continue

        t0 = time.time()
        try:
            sliding_embeddings = sliding_inference(str(local_path))
        except Exception as e:
            print(f"  SKIP (inference failed): {e}")
            if local_path.exists():
                local_path.unlink()
            done[key] = None
            save_progress(done)
            continue

        sw = sliding_embeddings.sliding_window
        n_windows = len(sliding_embeddings)

        all_embs = np.stack(
            [sliding_embeddings[wi].flatten() for wi in range(n_windows)])
        distances = cdist(ref_embedding, all_embs, metric="cosine")[0]
        similarities = 1.0 - distances

        for wi in range(n_windows):
            window = sw[wi]
            print(f"    window {wi:>3}/{n_windows}  "
                  f"({window.start:>7.1f}-{window.end:>7.1f}s)  "
                  f"similarity={similarities[wi]:.4f}")

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_window = sw[best_idx]

        elapsed = time.time() - t0

        if best_sim >= (1.0 - SIMILARITY_THRESHOLD):
            result = {
                "podcast_key": key,
                "wav_url": wav_url,
                "similarity": best_sim,
                "match_window": best_idx,
                "match_start": float(best_window.start),
                "match_end": float(best_window.end),
            }
            matches.append(result)
            done[key] = result
            print(f"  >>> MATCH  window {best_idx}/{n_windows} "
                  f"({best_window.start:.1f}-{best_window.end:.1f}s) "
                  f"similarity={best_sim:.4f}  [{elapsed:.1f}s]")
        else:
            done[key] = None
            print(f"  No match  (best similarity={best_sim:.4f}, "
                  f"threshold={1.0 - SIMILARITY_THRESHOLD:.2f})  [{elapsed:.1f}s]")

        if local_path.exists():
            local_path.unlink()

        save_progress(done)

    dl_pool.shutdown(wait=False)

    # ── Save results ───────────────────────────────────────────────────
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    MATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MATCHES_FILE, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"Speaker detected in {len(matches)} / {total_podcasts} podcasts")
    print(f"{'=' * 80}")
    if matches:
        print(f"\n{'#':<5}{'Similarity':<13}{'Time':>14}  {'Podcast'}")
        print("-" * 80)
        for rank, m in enumerate(matches, 1):
            time_str = f"{m['match_start']:.0f}-{m['match_end']:.0f}s"
            print(
                f"{rank:<5}{m['similarity']:<13.4f}{time_str:>14}  {m['podcast_key']}")
    print(f"\nResults written to {MATCHES_FILE}")


if __name__ == "__main__":
    main()

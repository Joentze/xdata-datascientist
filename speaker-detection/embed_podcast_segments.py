import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import chromadb
import requests
import torch
from pyannote.audio import Inference, Model

BASE_DIR = Path(
    "/Users/tanjoen/Documents/xdata-datascientist/speaker-detection")
SEGMENTS_JSON = BASE_DIR / "data" / "podcast_segments.json"
PODCASTS_DIR = BASE_DIR / "data" / "podcasts"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
PROGRESS_FILE = BASE_DIR / "data" / "embed_progress.json"
DOWNLOAD_TIMEOUT = 180
CHROMA_BATCH = 200
PREFETCH = 3
DOWNLOAD_WORKERS = 4
WINDOW_DURATION = 3.0
WINDOW_STEP = 1.0


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


def load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(done: set[str]) -> None:
    PROGRESS_FILE.write_text(json.dumps(sorted(done)))


def chroma_writer_loop(queue: Queue, collection: chromadb.Collection) -> None:
    """Background thread that drains the queue and writes batches to ChromaDB."""
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        ids, embeddings, metadatas = item
        try:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        except Exception as e:
            print(f"  ChromaDB write error: {e}")
        queue.task_done()


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: set the HF_TOKEN environment variable with your Hugging Face access token.")
        print("  1. Accept conditions at https://hf.co/pyannote/embedding")
        print("  2. Create a token at https://hf.co/settings/tokens")
        return

    with open(SEGMENTS_JSON) as f:
        podcasts: dict = json.load(f)

    total_podcasts = len(podcasts)
    print(f"Loaded {total_podcasts} podcasts")

    done_keys = load_progress()
    remaining = [(k, v) for k, v in podcasts.items() if k not in done_keys]
    if done_keys:
        print(
            f"Resuming: {len(done_keys)} already processed, {len(remaining)} remaining")
    if not remaining:
        print("All podcasts already processed.")
        return

    # ── Model ──────────────────────────────────────────────────────────
    print("Loading pyannote/embedding model...")
    model = Model.from_pretrained(
        "pyannote/embedding", use_auth_token=hf_token)
    inference = Inference(model, window="sliding",
                          duration=WINDOW_DURATION, step=WINDOW_STEP)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    inference.to(torch.device(device))
    print(f"Inference device: {device}")

    # ── ChromaDB ───────────────────────────────────────────────────────
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="speaker_embeddings",
        metadata={"hnsw:space": "cosine"},
    )
    print(
        f"ChromaDB collection at {CHROMA_DIR}  (existing: {collection.count()} embeddings)")

    # Background writer thread for non-blocking ChromaDB inserts
    write_queue: Queue = Queue(maxsize=20)
    writer = threading.Thread(
        target=chroma_writer_loop, args=(write_queue, collection), daemon=True,
    )
    writer.start()

    # ── Prefetch download pool ─────────────────────────────────────────
    PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
    dl_pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)

    download_futures: dict[str, Future] = {}
    for key, podcast in remaining[: PREFETCH]:
        local = podcast_key_to_local_path(key)
        download_futures[key] = dl_pool.submit(
            download_wav, podcast["wav_url"], local)

    # ── Main loop: sliding-window embed while next WAVs download ───────
    processed = len(done_keys)
    global_added = 0

    for i, (key, podcast) in enumerate(remaining):
        lookahead = i + PREFETCH
        if lookahead < len(remaining):
            nk, np_ = remaining[lookahead]
            if nk not in download_futures:
                download_futures[nk] = dl_pool.submit(
                    download_wav, np_[
                        "wav_url"], podcast_key_to_local_path(nk),
                )

        wav_url = podcast["wav_url"]
        local_path = podcast_key_to_local_path(key)
        processed += 1

        print(f"\n[{processed}/{total_podcasts}] {key}")

        try:
            if key in download_futures:
                download_futures.pop(key).result()
            else:
                download_wav(wav_url, local_path)
        except Exception as e:
            print(f"  SKIP (download failed): {e}")
            continue

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        added = 0

        def flush():
            nonlocal ids, embeddings, metadatas, added
            if not ids:
                return
            write_queue.put((ids[:], embeddings[:], metadatas[:]))
            added += len(ids)
            ids, embeddings, metadatas = [], [], []

        t0 = time.time()
        try:
            sliding_embeddings = inference(str(local_path))
        except Exception as e:
            print(f"  SKIP (inference failed): {e}")
            if local_path.exists():
                local_path.unlink()
            continue

        sw = sliding_embeddings.sliding_window
        n_windows = len(sliding_embeddings)
        print(f"  {n_windows} windows (duration={sw.duration:.1f}s, step={sw.step:.1f}s)")

        for wi in range(n_windows):
            window = sw[wi]
            emb = sliding_embeddings[wi]
            start, end = window.start, window.end

            window_id = f"{key}__w{wi}"
            ids.append(window_id)
            embeddings.append(emb.flatten().tolist())
            metadatas.append({
                "start_time": start,
                "end_time": end,
                "podcast_key": key,
                "wav_url": wav_url,
                "window_index": wi,
            })

            if len(ids) >= CHROMA_BATCH:
                flush()

        flush()

        elapsed = time.time() - t0
        global_added += added
        if added:
            print(f"  Embedded {added} windows in {elapsed:.1f}s")
        else:
            print("  No embeddings extracted")

        if local_path.exists():
            local_path.unlink()

        done_keys.add(key)
        save_progress(done_keys)

    # drain remaining ChromaDB writes
    write_queue.put(None)
    writer.join()
    dl_pool.shutdown(wait=False)

    print(f"\nDone. Added {global_added} embeddings this run.")
    print(f"Total in ChromaDB: {collection.count()}")


if __name__ == "__main__":
    main()

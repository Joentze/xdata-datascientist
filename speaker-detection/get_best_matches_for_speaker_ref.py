import json
import os
from pathlib import Path

import chromadb
import numpy as np
import torch
from pyannote.audio import Inference, Model

BASE_DIR = Path(
    "/Users/tanjoen/Documents/xdata-datascientist/speaker-detection")
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
SPEAKER_REF = BASE_DIR / "speaker_ref.wav"
OUTPUT_JSON = BASE_DIR / "data" / "speaker_matches.json"

TOP_K = 1000
DISTANCE_THRESHOLD = 0


def embed_reference(inference: Inference, wav_path: Path) -> np.ndarray:
    """Embed the entire reference WAV as a single utterance."""
    embedding = inference(str(wav_path))
    return np.array(embedding).flatten()


def dedupe_by_podcast(ids, metadatas, distances) -> list[dict]:
    """Keep only the best (lowest distance) hit per podcast."""
    best: dict[str, dict] = {}
    for id_, meta, dist in zip(ids, metadatas, distances):
        key = meta["podcast_key"]
        if key not in best or dist < best[key]["distance"]:
            best[key] = {"id": id_, "distance": dist, **meta}
    return sorted(best.values(), key=lambda x: x["distance"])


def main():
    if not SPEAKER_REF.exists():
        print(f"Reference WAV not found: {SPEAKER_REF}")
        return

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: set the HF_TOKEN environment variable with your Hugging Face access token.")
        print("  1. Accept conditions at https://hf.co/pyannote/embedding")
        print("  2. Create a token at https://hf.co/settings/tokens")
        return

    # ── Model ──────────────────────────────────────────────────────────
    print("Loading pyannote/embedding model...")
    model = Model.from_pretrained(
        "pyannote/embedding", use_auth_token=hf_token)
    inference = Inference(model, window="whole")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    inference.to(torch.device(device))
    print(f"Inference device: {device}")

    # ── Embed reference ────────────────────────────────────────────────
    print(f"Embedding reference: {SPEAKER_REF.name}")
    ref_embedding = embed_reference(inference, SPEAKER_REF)
    print(f"Reference embedding shape: {ref_embedding.shape}")

    # ── ChromaDB query ─────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name="speaker_embeddings")
    total = collection.count()
    print(f"ChromaDB collection has {total} embeddings")

    print(f"Querying top {TOP_K} nearest neighbours...")
    results = collection.query(
        query_embeddings=[ref_embedding.tolist()],
        n_results=TOP_K,
        include=["metadatas", "distances"],
    )

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    all_matches = dedupe_by_podcast(ids, metadatas, distances)
    matches = [m for m in all_matches if m["distance"] <= DISTANCE_THRESHOLD]
    print(f"Found {len(matches)} distinct podcasts within threshold {DISTANCE_THRESHOLD} "
          f"(from {len(all_matches)} deduped, {len(ids)} window hits)")

    # ── Display ────────────────────────────────────────────────────────
    print(f"\n{'Rank':<6}{'Distance':<12}{'Podcast Key':<60}{'Time':>10}")
    print("-" * 90)
    for rank, m in enumerate(matches, 1):
        start = m.get("start_time", 0)
        end = m.get("end_time", 0)
        time_str = f"{start:.1f}-{end:.1f}s"
        print(f"{rank:<6}{m['distance']:<12.4f}{m['podcast_key']:<60}{time_str:>10}")

    # ── Save ───────────────────────────────────────────────────────────
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

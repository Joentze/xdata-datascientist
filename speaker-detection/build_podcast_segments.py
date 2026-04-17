import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

AUDIO_BASE = "https://a3s.fi/swift/v1/YCSEP_v2"
INPUT_CSV = Path(__file__).resolve().parent / ".." / "asr" / "data" / "YCSEP_static.csv"
OUTPUT_JSON = Path(__file__).resolve().parent / "data" / "podcast_segments.json"


def audio_url_to_podcast_key(audio_url: str) -> str:
    """Strip the _start_end timestamp suffix to get a unique podcast identifier."""
    segment_id = unquote(urlparse(audio_url).path.rstrip("/").split("/")[-1])
    return re.sub(r"_\d+_\d+$", "", segment_id)


def podcast_key_to_wav_url(key: str) -> str:
    """Convert a podcast key to the remote WAV URL (same logic as download_podcasts.py)."""
    match = re.search(r"(\d{8}--)", key)
    slug = key[match.start():]
    date, video_id, title_slug = slug.split("--", 2)
    title = title_slug.replace("-", " ")
    return f"{AUDIO_BASE}/{date}--{video_id}--{quote(title)}.wav"


def main():
    podcasts: dict[str, list[dict]] = defaultdict(list)

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_url = row["audio"].strip()
            if not audio_url:
                continue
            key = audio_url_to_podcast_key(audio_url)
            start = float(row["start_time"])
            end = float(row["end_time"])
            podcasts[key].append({"start": start, "end": end, "audio": audio_url})

    result = {}
    for key, segments in podcasts.items():
        segments.sort(key=lambda s: s["start"])
        result[key] = {
            "wav_url": podcast_key_to_wav_url(key),
            "segments": segments,
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Found {len(result)} distinct podcasts")
    print(f"Total segments: {sum(len(v['segments']) for v in result.values())}")
    print(f"Output written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

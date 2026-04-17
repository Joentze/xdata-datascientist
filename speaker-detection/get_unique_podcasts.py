import re
import csv
from pathlib import Path

INPUT_CSV = Path(__file__).resolve().parent.parent / "asr" / "data" / "YCSEP_static.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_CSV = OUTPUT_DIR / "unique_podcasts.csv"

TIMESTAMP_SUFFIX = re.compile(r"_\d+_\d+$")

unique_episodes = set()

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        audio_url = row["audio"].strip()
        if not audio_url:
            continue
        episode = TIMESTAMP_SUFFIX.sub("", audio_url)
        unique_episodes.add(episode)

episodes = sorted(unique_episodes)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["episode_url"])
    for ep in episodes:
        writer.writerow([ep])

print(f"Found {len(episodes)} unique podcast episodes → {OUTPUT_CSV}")

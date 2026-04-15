import io
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import requests
from pydub import AudioSegment

AUDIO_BASE = "https://a3s.fi/swift/v1/YCSEP_v2"
CLIP_API = "https://ycsep-fastapi-viewer-ycsep-test.2.rahtiapp.fi/clip"


def parse_viewer_url(viewer_url: str) -> str:
    segment_id = unquote(urlparse(viewer_url).path.rstrip("/").split("/")[-1])

    body = re.sub(r"_\d+_\d+$", "", segment_id)

    match = re.search(r"(\d{8}--)", body)
    filename_slug = body[match.start():]

    date, video_id, title_slug = filename_slug.split("--", 2)
    title = title_slug.replace("-", " ")

    return f"{AUDIO_BASE}/{date}--{video_id}--{quote(title)}.wav"


def audio_filename(url: str) -> str:
    stem = Path(unquote(urlparse(url).path).split("/")[-1]).stem
    return f"{stem}.mp3"


def download_audio(
    url: str,
    output_dir: str | Path = "audio",
    timeout: int = 60,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / audio_filename(url)

    if dest.exists():
        print(f"Already exists, skipping: {dest.name}")
        return dest

    print(f"Downloading full audio: {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    audio = AudioSegment.from_file(io.BytesIO(resp.content), format="wav")
    audio.export(dest, format="mp3")
    print(f"Saved: {dest}")
    return dest


def clip_filename(url: str, start: float, end: float) -> str:
    stem = Path(unquote(urlparse(url).path).split("/")[-1]).stem
    return f"{stem}_{int(start)}_{int(end)}.wav"


def download_clip(
    url: str,
    start: float,
    end: float,
    output_dir: str | Path = "audio",
    timeout: int = 60,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / clip_filename(url, start, end)

    if dest.exists():
        print(f"Already exists, skipping: {dest.name}")
        return dest

    print(f"Downloading clip: {url} [{start}–{end}]")
    resp = requests.get(
        CLIP_API,
        params={"url": url, "start": start, "end": end, "fmt": "wav"},
        timeout=timeout,
    )
    resp.raise_for_status()

    dest.write_bytes(resp.content)
    print(f"Saved: {dest}")
    return dest


if __name__ == "__main__":
    viewer_url = "https://ycsep-fastapi-viewer-ycsep-test.2.rahtiapp.fi/audio/Singapore.Historyogi.20240505--emPCRSs7f6k--Historyogi-Podcast-EP4%EF%BC%9A-The-Beginner%27s-Guide-to-the-South-China-Sea-dispute_159725_177917"

    url = parse_viewer_url(viewer_url)
    download_audio(url)

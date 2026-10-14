#!/usr/bin/env python3
"""Extract complete MP3 audio from the original video, with a verified cache."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .extract_frames import cache_lock
from .release_tools import sha

PROTOCOL = "omnicompass-full-audio-container-clock-v1"
ENCODING = {"format": "mp3", "sample_rate": 16000, "channels": 1, "bitrate": 32000}


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise ValueError(f"{Path(args[0]).name} failed: {result.stderr[-2000:]}")
    return result.stdout


def probe(path):
    import av

    with av.open(str(path)) as container:
        streams = [
            {
                "index": s.index,
                "codec_name": s.codec_context.name,
                "sample_rate": s.codec_context.sample_rate,
                "channels": s.codec_context.channels,
                "start_time": float(s.start_time * s.time_base) if s.start_time is not None else None,
            }
            for s in container.streams.audio
        ]
        return {
            "streams": streams,
            "origin": float(container.start_time / av.time_base) if container.start_time is not None else 0,
            "duration": float(container.duration / av.time_base) if container.duration is not None else None,
        }


def extract_audio(video, cache_dir):
    ffmpeg = shutil.which(os.environ.get("FFMPEG_BINARY", "ffmpeg"))
    if not ffmpeg:
        raise ValueError("Audio extraction requires ffmpeg on PATH or FFMPEG_BINARY")
    video = Path(video).resolve(strict=True)
    stat = video.stat()
    identity = {
        "protocol": PROTOCOL,
        "source_sha256": sha(video),
        "extractor_sha256": sha(Path(__file__)),
        "encoding": ENCODING,
        "ffmpeg_version": command([ffmpeg, "-version"]),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / key
    with cache_lock(root, key):
        if destination.exists():
            try:
                manifest = json.loads((destination / "audio.json").read_text())
                audio = destination / "audio.mp3"
                if destination.is_symlink() or audio.is_symlink() or manifest["identity"] != identity or not audio.is_file() or sha(audio) != manifest["sha256"]:
                    raise ValueError("Audio cache identity/content mismatch")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid audio cache {destination}: {exc}. Use a new cache directory or remove only this entry.") from exc
            return audio, manifest, True
        source = probe(video)
        streams = source["streams"]
        if not streams:
            raise ValueError(f"No audio stream in {video.name}; select --audio off for a vision-only run, or select videos with audio.")
        origin = source["origin"]
        if not math.isfinite(origin):
            raise ValueError("Invalid source timeline origin")
        stage = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
        try:
            audio = stage / "audio.mp3"
            # Keep audio on the same container-relative clock as sampled frames.
            # Leading silence represents a delayed real stream, not missing audio.
            command(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-copyts",
                    "-start_at_zero",
                    "-i",
                    str(video),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-af",
                    "aresample=async=1:first_pts=0",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "32k",
                    str(audio),
                ]
            )
            output = probe(audio)
            stream = output["streams"][0]
            duration = output["duration"]
            if not audio.stat().st_size or not math.isfinite(duration) or duration <= 0 or stream["codec_name"] not in {"mp3", "mp3float"} or int(stream["sample_rate"]) != 16000 or stream["channels"] != 1:
                raise ValueError("Invalid extracted audio")
            current = video.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (current.st_size, current.st_mtime_ns, current.st_ino):
                raise ValueError("Source video changed during audio extraction")
            manifest = {
                "identity": identity,
                "sha256": sha(audio),
                "bytes": audio.stat().st_size,
                "duration_seconds": duration,
                "source_audio_stream": streams[0]["index"],
                "timeline_origin_seconds": origin,
                "source_audio_start_seconds": streams[0]["start_time"],
            }
            (stage / "audio.json").write_text(json.dumps(manifest, indent=2) + "\n")
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return destination / "audio.mp3", manifest, False


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--video", type=Path, required=True)
    cli.add_argument("--cache-dir", type=Path, required=True)
    args = cli.parse_args()
    audio, manifest, hit = extract_audio(args.video, args.cache_dir)
    print(json.dumps({"audio": str(audio), "cache_hit": hit, **manifest}, indent=2))


if __name__ == "__main__":
    main()

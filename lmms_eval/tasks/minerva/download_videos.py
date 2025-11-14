#!/usr/bin/env python3
"""
Script to download YouTube videos from minerva.json dataset.
Downloads unique videos only, skipping duplicates and already downloaded files.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Set
from tqdm import tqdm


def load_video_ids(json_path: Path) -> list:
    """Load all video IDs from the JSON file."""
    print(f"Loading video IDs from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    video_ids = [item['video_id'] for item in data if 'video_id' in item]
    print(f"Found {len(video_ids)} total entries")
    return video_ids

def get_unique_video_ids(video_ids: list) -> list:
    """Get unique video IDs while preserving order."""
    seen = set()
    unique_ids = []
    for vid in video_ids:
        if vid not in seen:
            seen.add(vid)
            unique_ids.append(vid)
    print(f"Found {len(unique_ids)} unique video IDs")
    return unique_ids

def get_already_downloaded(output_dir: Path) -> Set[str]:
    """Check which videos have already been downloaded."""
    if not output_dir.exists():
        return set()
    
    downloaded = set()
    # Common video extensions
    extensions = ['.mp4', '.webm', '.mkv', '.flv', '.avi', '.mov']
    
    for file in output_dir.iterdir():
        if file.is_file() and file.suffix in extensions:
            # Extract video ID from filename (assumes format: video_id.ext)
            video_id = file.stem
            downloaded.add(video_id)
    
    print(f"Found {len(downloaded)} already downloaded videos")
    return downloaded

def download_video(video_id: str, output_dir: Path) -> bool:
    """Download a single YouTube video using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    if os.path.exists(output_template):
        return True
    
    try:
        # Using yt-dlp for downloading
        # -f best: download best quality
        # --no-playlist: don't download playlists
        # -o: output template
        cmd = [
            "yt-dlp",
            "-f", "best",
            "--no-playlist",
            "-o", output_template,
            "--cookies", "cookies.txt",
            url
        ]
        print('cmd', ' '.join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return True
        else:
            print(f"✗ Failed to download {video_id}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout while downloading {video_id}")
        return False
    except FileNotFoundError:
        print("Error: yt-dlp is not installed. Please install it with:")
        print("  pip install yt-dlp")
        print("  or: sudo apt-get install yt-dlp")
        raise
    except Exception as e:
        print(f"✗ Error downloading {video_id}: {e}")
        return False

def main(output_dir: Path, json_file: Path):
    """Main function to orchestrate video downloads."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load video IDs from JSON
    all_video_ids = load_video_ids(json_file)
    
    # Get unique video IDs
    unique_video_ids = get_unique_video_ids(all_video_ids)
    
    # Check which videos are already downloaded
    already_downloaded = get_already_downloaded(output_dir)
    
    # Filter out already downloaded videos
    to_download = [vid for vid in unique_video_ids if vid not in already_downloaded]
    
    if not to_download:
        print("\nAll videos are already downloaded!")
        return
    
    print(f"\nNeed to download {len(to_download)} videos")
    print(f"Skipping {len(already_downloaded)} already downloaded videos\n")
    
    # Download videos
    successful = 0
    failed = 0
    
    for i, video_id in enumerate(tqdm(to_download, desc="Downloading videos", total=len(to_download)), 1):
        print(f"\n[{i}/{len(to_download)}] Processing {video_id}")
        if download_video(video_id, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total unique videos: {len(unique_video_ids)}")
    print(f"Already downloaded: {len(already_downloaded)}")
    print(f"Attempted downloads: {len(to_download)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*60)

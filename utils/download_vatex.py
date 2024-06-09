import subprocess
import json
from tqdm import tqdm

with open("data/vatex_public_test_english_v1.1.json",'r')as f:
    testvideo=json.load(f)
for video in tqdm(testvideo):
    video_id = video['videoID']
    command = f"yt-dlp -o data/test_video/{video_id}.mp4 -f mp4 https://www.youtube.com/watch?v={video_id}"
    subprocess.run(command, shell=True)
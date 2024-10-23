import json
import os

import av
from tqdm import tqdm

data_stats = {
    "length_distribution": {
        "(0, 2]_min": 0,
        "(2, 3]_min": 0,
        "(3, 4]_min": 0,
        "(4, 6]_min": 0,
        "(6, 8]_min": 0,
        "(8, 10]_min": 0,
        "(10, 12]_min": 0,
        "(12, 15]_min": 0,
        "(15)_min": 0,
    },
    "average_length_seconds": 0,
    "average_length_minutes": 0,
    "total_files": 0,
}


# This one is faster
def record_video_length_stream(container):
    video = container.streams.video[0]
    video_length = float(video.duration * video.time_base)  # in seconds
    return video_length


# This one works for all types of video
def record_video_length_packet(container):
    video_length = 0
    for packet in container.demux(video=0):
        for frame in packet.decode():
            video_length = frame.time  # The last frame time represents the video time
    return video_length


if __name__ == "__main__":
    directory = "/mnt/sfs-common/krhu/.cache/huggingface/Combined_milestone/videos"
    total_length = 0
    mp4_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]
    data_stats["total_files"] = len(mp4_files)

    for video_file in tqdm(mp4_files, desc="Processing videos"):
        video_path = os.path.join(directory, video_file)
        container = av.open(video_path)

        if "webm" not in video_path and "mkv" not in video_path:
            try:
                video_length = record_video_length_stream(container)  # in seconds
            except:
                video_length = record_video_length_packet(container)
        else:
            video_length = record_video_length_packet(container)

        total_length += video_length

        # Count length distribution
        video_length_minutes = video_length / 60
        if 0 < video_length_minutes <= 2:
            data_stats["length_distribution"]["(0, 2]_min"] += 1
        elif 2 < video_length_minutes <= 3:
            data_stats["length_distribution"]["(2, 3]_min"] += 1
        elif 3 < video_length_minutes <= 4:
            data_stats["length_distribution"]["(3, 4]_min"] += 1
        elif 4 < video_length_minutes <= 6:
            data_stats["length_distribution"]["(4, 6]_min"] += 1
        elif 6 < video_length_minutes <= 8:
            data_stats["length_distribution"]["(6, 8]_min"] += 1
        elif 8 < video_length_minutes <= 10:
            data_stats["length_distribution"]["(8, 10]_min"] += 1
        elif 10 < video_length_minutes <= 12:
            data_stats["length_distribution"]["(10, 12]_min"] += 1
        elif 12 < video_length_minutes <= 15:
            data_stats["length_distribution"]["(12, 15]_min"] += 1
        else:
            data_stats["length_distribution"]["(15)_min"] += 1

    # Calculate average length
    if data_stats["total_files"] > 0:
        average_length_seconds = total_length / data_stats["total_files"]
        average_length_minutes = average_length_seconds / 60
        data_stats["average_length_seconds"] = average_length_seconds
        data_stats["average_length_minutes"] = average_length_minutes

    with open("./video_length_stats.json", "w") as f:
        json.dump(data_stats, f, indent=4, ensure_ascii=False)

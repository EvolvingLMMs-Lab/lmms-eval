import os


def get_cache_dir(config, sub_dir="videos"):
    HF_HOME = os.environ["HF_HOME"]
    cache_dir = config["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(HF_HOME, cache_dir)
    cache_dir = os.path.join(cache_dir, sub_dir)
    return cache_dir


def _get_video_file(prefix: str, video_name: str, suffix: str):
    if not video_name.endswith(suffix):
        video_name = f"{video_name}.{suffix}"
    video_path = os.path.join(prefix, video_name)
    return video_path


def get_video(prefix: str, video_name: str, suffix: str = "mp4"):
    tried = [os.path.abspath(_get_video_file(prefix, video_name, suffix)), os.path.abspath(_get_video_file(prefix, video_name, suffix.upper())), os.path.abspath(_get_video_file(prefix, video_name, suffix.lower()))]
    for video_path in tried:
        if os.path.exists(video_path):
            return video_path
    raise FileNotFoundError(f"Tried both {tried} but none of them exist, please check")

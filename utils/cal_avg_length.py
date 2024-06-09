import os
import datetime
import cv2

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

def calculate_average_duration(directory):
    total_duration = 0
    video_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            video_path = os.path.join(directory, filename)
            duration = get_video_duration(video_path)
            total_duration += duration
            video_count += 1
    if video_count > 0:
        average_duration = total_duration / video_count
        return average_duration
    else:
        return 0

directory = 'data/test_video'

# calculate average duration
average_duration = calculate_average_duration(directory)


print(f"Avg duration of '{directory}' : {datetime.timedelta(seconds=average_duration)}")
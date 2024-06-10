import json
from lmms_eval.tasks import initialize_tasks, include_path, get_task_dict
import av
from tqdm import tqdm
from av.codec.context import CodecContext

tasks = ["worldqa_gen", "activitynetqa", "nextqa_oe_val", "nextqa_oe_test", "videochatgpt_gen", "egoschema"]
# tasks = ["nextqa_oe_val"]
data_stats = {}

# This one is faster
def record_video_length_stream(container):
    video = container.streams.video[0]
    video_length = (float(video.duration * video.time_base)) # in seconds
    return video_length

# This one works for all types of video
def record_video_length_packet(container):
    video_length = 0
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            video_length = frame.time # The last frame time represent the video time  
    
    return video_length

if __name__ == "__main__":
    initialize_tasks()

    task_dict = get_task_dict(tasks, model_name="llavavid")
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if type(task_obj) == tuple:
            group, task_obj = task_obj
        
        docs = task_obj.test_docs()
        doc_to_visual = task_obj.doc_to_visual
        data_stats[task_name] = 0
        for doc in tqdm(docs, desc=f"Processing {task_name}"):
            video_path = doc_to_visual(doc)
            container = av.open(video_path[0])
            
            if "webm" not in video_path[0] and "mkv" not in video_path[0]:
                try:
                    video_length = record_video_length_stream(container) # in seconds
                except:
                    video_length = record_video_length_packet(container)
            else:
                video_length = record_video_length_packet(container)
            data_stats[task_name] += video_length
        
        data_stats[task_name] /= len(docs) # into seconds
        # data_stats[task_name] /= 60 # into minutes
    
    with(open("./video_benchmarks_stats.json", "w")) as f:
        json.dump(data_stats, f, indent=4, ensure_ascii=False)


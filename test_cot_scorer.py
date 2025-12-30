# /workspace/lmms-eval-miao/test_cot_scorer.py
from lmms_eval.models.model_utils import FrameCoTScorer

video_path = "/workspace/.hf_home/videomme/data/fFjv93ACGo8.mp4"
query = "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?"


scorer = FrameCoTScorer(
    scorer_type=2,
    reasoner_name="Qwen/Qwen2-VL-7B-Instruct",
    scorer_name="Qwen/Qwen2-VL-7B-Instruct",
    candidates=8,
    task_name="videomme_debug",
    device="cuda:0",
    use_cache=False,
    debug_print=False,
)

out = scorer.score_video(video_path=video_path, query=query)
print(out["meta"])
print(list(out["frames"].items())[:8])

scorer2 = FrameCoTScorer(
    scorer_type=2,
    reasoner_name="lmms-lab/llava-onevision-qwen2-7b-ov",
    scorer_name=  "lmms-lab/llava-onevision-qwen2-7b-ov",   
    candidates=8,
    task_name="videomme_debug",
    device="cuda:1",
    use_cache=True,
    debug_print=False,)
out2 = scorer2.score_video(video_path=video_path, query=query)
print(out2["meta"])
print(list(out2["frames"].items())[:8]) 

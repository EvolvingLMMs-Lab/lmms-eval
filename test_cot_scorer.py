# /workspace/lmms-eval-miao/test_cot_scorer.py
from lmms_eval.models.model_utils import FrameCoTScorer

video_path = "/workspace/.hf_home/videomme/data/fFjv93ACGo8.mp4"
query = "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?", "A. Apples.","B. Candles.","C. Berries.","D. The three kinds are of the same number."


scorer = FrameCoTScorer(
    scorer_type=2,
    reasoner_name="Qwen/Qwen2-VL-7B-Instruct",
    scorer_name="Qwen/Qwen2-VL-7B-Instruct",
    candidates=8,
    task_name="videomme_debug",
    device="cuda:0",
    use_cache=False,
    debug_print=True,
)

out = scorer.score_video(video_path=video_path, query=query)
print(out["meta"])
print(list(out["frames"].items())[:8])

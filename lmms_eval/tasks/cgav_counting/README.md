# CG-AV-Counting

This directory implements all three protocols from the official
[CG-AV-Counting implementation](https://github.com/open-compass/VLMEvalKit/tree/main/vlmeval/dataset/CGAVCounting):

- `cgav_counting_long`: full-video count (ACC, OBOA, MAE, RMSE)
- `cgav_counting_ref`: reference-clip count (ACC, OBOA, MAE, RMSE)
- `cgav_counting_clue`: white-box clue grounding (WCS, IFA)
- `cgav_counting`: group that runs all three

The dataset is gated and large. Accept its terms at
https://huggingface.co/datasets/CG-Bench/CG-AV-Counting and configure a Hugging
Face token. Normal task execution downloads the dataset, then automatically merges
and extracts the official split `videos.zip.part*` and `ref_videos.zip.part*`
archives on first use. The media cache is `$HF_HOME/cgav_counting` by default;
set `CGAV_COUNTING_ROOT` only to reuse a different existing cache.

```bash
accelerate launch -m lmms_eval --model <model> --tasks cgav_counting_long --batch_size 1
```

Audio-dependent query types (`A`, `A2V`, `V2A`, and `AV`) require a model/backend
that preserves the video's audio track.

# CG-Bench

This task ports the black-box multiple-choice evaluation from the
[VideoITG lmms-eval integration](https://github.com/NVlabs/VideoITG/tree/main/lmms_eval/tasks/cgbench)
and the official [CG-Bench](https://github.com/CG-Bench/CG-Bench) protocol.
`cgbench` evaluates video only;
`cgbench_subtitles` adds subtitles aligned to the model's uniformly sampled frames;
`cgbench_all` runs both variants. The default is the official 3,000-question mini split.

The dataset is gated. Accept its terms at
https://huggingface.co/datasets/CG-Bench/CG-Bench and authenticate with a Hugging
Face token before running. Media is normally downloaded and extracted under
`$HF_HOME/cg_videos_720p`. Existing media can instead be exposed through
`CGBENCH_VIDEO_DIR`; subtitles can be exposed through `CGBENCH_SUBTITLE_DIR`.

```bash
accelerate launch -m lmms_eval --model <model> --tasks cgbench --batch_size 1
```

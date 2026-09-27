# Video-Index

[Video-Index](https://huggingface.co/datasets/GMLRVigil/Video-Index) holds 840 multiple-choice
video questions drawn from 76 public video benchmarks: one question per video and 210 questions
in each of four capability groups (perception, temporal, spatial / physical, reasoning /
knowledge). Every item passed a screen with text-only, single-frame, options-only and
shuffled-frame attackers, and its marked answer was verified against the frames. Items have 2 to
10 options; the mean chance accuracy is 29.7.

The dataset is public. The item file is loaded with `datasets`; each video is downloaded from
the dataset repository when its item is first used, so a run with `--limit` downloads only the
videos of the selected items. Set `VIDEO_INDEX_DIR` to a local copy of the repository (the
directory that holds `videos/`) to read the videos from disk.

```bash
python -m lmms_eval --model <model> --tasks video_index --batch_size 1
python -m lmms_eval --model <model> --tasks video_index_blind --batch_size 1
```

## Tasks

| Task | Visual input | Documents |
| --- | --- | --- |
| `video_index` | The video file; the model wrapper applies its own frame sampling. | 840 |
| `video_index_1fps` | Frames of the paper protocol as images: one frame per second of the timeline, at most 512 frames (uniform thinning beyond), short side 224 pixels. | 840 |
| `video_index_64frame`, `video_index_32frame`, `video_index_8frame` | The same frame rule with a cap of 64, 32 or 8 frames. | 840 |
| `video_index_blind` | None: the question and the options only, under four option permutations per item. | 3,360 |

The frame tasks reproduce the frames of the paper protocol independently of the model wrapper.
They suit wrappers that accept a list of images; use `video_index` for wrappers that read video
files, and report the frame budget of the wrapper with the score. The frames are decoded in
stream order, because random access returns a neighbouring frame on some of the videos.

The paper sent the frames as JPEG images (quality 85). Wrappers that build OpenAI-style messages
encode images as PNG by default; `LMMS_IMAGE_ENCODE_FORMAT=JPEG` selects the encoding of the
paper. On a 12-item check with `gemini-2.5-flash-lite` and 8 frames, the JPEG setting returned
the replies of the reference runner on 12 of 12 items and the PNG default on 9 of 12.

The prompt is the one of the paper:

```text
You are given {n} frame(s) sampled from a video. Answer the question based on these frames.

Question: {question}
A. {option}
B. {option}
Reply with ONLY the option letter (or the exact short answer if no options).
```

`video_index` replaces the first sentence by `You are given a video. Answer the question based
on this video.` because the task does not know how many frames the wrapper samples.
`video_index_blind` uses `You are given NO frames from the video. Answer the question from the
text alone.` The option permutations of the blind task are fixed per item
(`random.Random("42|<item_id>")`), so every model sees the same four orders.

## Metrics

| Metric | Definition |
| --- | --- |
| `video_index_acc` | Accuracy in percent over all items. |
| `video_index_perception`, `video_index_temporal`, `video_index_spatial`, `video_index_reasoning` | Accuracy in percent over the items of one capability group. |

Scoring is rule based and needs no judge model. The option named by a reply is, in this order,
the leading option letter (`B`, `(B)`, `B. text`), a stated answer (`the answer is B`; the last
statement counts), or the single option whose text the reply repeats. A reply that names no
option is scored as wrong. In the blind task the four replies of an item are averaged before the
mean over items is taken. The gain of a model is its accuracy on a video task minus its accuracy
on `video_index_blind`. A capability group without items in the evaluated slice (`--limit`)
reports `nan`.

## Reference values

Claude Opus 5 with the paper protocol (`video_index_1fps` setting): 56.8 overall, 21.3 blind.
`generation_kwargs` sets `max_new_tokens: 16`, the value of the paper for models that reply
directly; models that reason before the reply need a larger value.

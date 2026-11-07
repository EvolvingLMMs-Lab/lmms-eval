# IMAVB

IMAVB tests whether an omnimodal model catches a question whose premise contradicts what it sees or hears in a long movie clip.

- Paper: [Senses Wide Shut: A Representation-Action Gap in Omnimodal LLMs](https://arxiv.org/abs/2605.13737)
- Dataset: [ngqtrung/IMAVB](https://huggingface.co/datasets/ngqtrung/IMAVB) (500 clips, 60-300 s, video with audio, 10.9 GB)
- Code: [ngquangtrung57/imavb](https://github.com/ngquangtrung57/imavb)

## Tasks

Each clip has four questions, one in each split (500 items each). All items are six-way multiple choice: A-D are content answers, E is "The visual detail in the question is incorrect", F is "The audio detail in the question is incorrect".

| Task | Split | Correct answer |
|---|---|---|
| `imavb_standard_vision` | standard premise, vision query | A-D |
| `imavb_standard_audio` | standard premise, audio query | A-D |
| `imavb_misleading_vision` | one visual premise detail swapped | E |
| `imavb_misleading_audio` | one audio premise detail swapped | F |
| `imavb` (group) | all four | mean of the four split accuracies = balanced accuracy (Bal) |

The metric is accuracy. The `imavb` group score is the unweighted mean of the four split accuracies, which equals the balanced accuracy of the paper, `Bal = ((std_v + std_a) / 2 + (mis_v + mis_a) / 2) / 2`.

## Protocol

The task uses the A1 baseline prompt of the paper with the options in fixed order, greedy decoding and `max_new_tokens: 16`. The answer letter is read with the shared `extract_mcq_answer` helper; when it abstains, a response that quotes exactly one option text maps to that option.

`doc_to_visual` returns the clip path, so the model must load the audio track of the video. In the paper, every model receives 50 uniformly sampled frames and the full audio track at 16 kHz mono. Set the frame count with the model arguments of your backend. Reasoning models need a larger budget, for example `--gen_kwargs max_new_tokens=4096`.

## Usage

```bash
python -m lmms_eval \
    --model qwen2_5_omni --force_simple \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B,attn_implementation=sdpa \
    --tasks imavb \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
```

The first run downloads the dataset repository, including the videos, and links it to `$HF_HOME/IMAVB`. Check that your backend sends the audio track of the video to the model: for `qwen2_5_omni`, the simple adapter (`--force_simple`) reads the audio of video inputs, and the chat adapter does not.

The clips are long, so the default video budget of `qwen2_5_omni` can run out of memory on an 80 GB GPU. To use a smaller budget, set `MODEL_SEQ_LEN=32768` in the environment (read by `qwen-omni-utils`).

## Citation

```bibtex
@inproceedings{nguyen2026senses,
  title     = {Senses Wide Shut: A Representation--Action Gap in Omnimodal {LLM}s},
  author    = {Nguyen Quang, Trung and Gao, Yiming and Pu, Fanyi and Zhang, Kaichen and Sun, Shuo and Liu, Ziwei},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

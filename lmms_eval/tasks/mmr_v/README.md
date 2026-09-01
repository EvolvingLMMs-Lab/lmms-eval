# MMR-V

[MMR-V: Can MLLMs Think with Video? A Benchmark for Multimodal Deep Reasoning in Videos](https://arxiv.org/abs/2506.04141)
asks models to mine evidence across long-range, non-adjacent frames instead of matching the
frame named in the question. The test split holds 1,257 multiple-choice questions over 317
videos, with 7 to 13 options per question (letters A-L), 10 `abilityType_L2` categories and
6 `videoType` categories.

`mmr_v` uses the official direct-answer prompt; `mmr_v_cot` uses the official chain-of-thought
prompt (`--with_cot` in the reference implementation) and reads the answer back from the
`[[X]]` block; `mmr_v_all` runs both. Scoring is rule-based exact match on the option letter,
so no LLM judge is needed.

Videos ship as the split archives `videos.tar.part.aa` .. `videos.tar.part.av` in
[JokerJan/MMR-VBench](https://huggingface.co/datasets/JokerJan/MMR-VBench). lmms-eval
concatenates and extracts them under `$HF_HOME/mmr_v` on the first run. Media that is already
on disk can instead be exposed through `MMR_V_VIDEO_DIR` (or `MMR_V_ROOT`); a `videos/` or
`videos_extracted/` subdirectory below that root is searched as well.

```bash
accelerate launch -m lmms_eval --model <model> --tasks mmr_v --batch_size 1
```

Official reference numbers for the overall accuracy, for a sanity target:
Gemini-2.5-pro 64.3, o4-mini 52.5, Gemini-2.5-Flash 51.2, human 86.0.

One upstream data defect is kept as-is: question 268 (`A Single Life - Oscar Nominated
Animated Short.mp4`) records `(K)` as the correct answer but offers only the options A-J, so
that question always scores 0.

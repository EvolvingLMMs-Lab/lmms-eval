# MDPBench

MDPBench evaluates multilingual document-to-Markdown recognition for digital
and photographed pages. The task reports the benchmark's text edit distance,
formula CDM, table TEDS, and overall scores, including language and acquisition
type breakdowns.

The task uses the public Hugging Face dataset
[`Delores-Lin/MDPBench-VLMEvalKit`](https://huggingface.co/datasets/Delores-Lin/MDPBench-VLMEvalKit).
Its samples are exposed through the dataset's `train` split, which is therefore
used as the evaluation split in `mdpbench.yaml`.

## Installation

Install the task-specific Python dependencies:

```bash
pip install -r lmms_eval/tasks/mdpbench/requirements.txt
```

Formula scoring also requires the system commands used by the reference CDM
implementation:

- Node.js 16 or newer
- XeLaTeX with the `xeCJK` package and Source Han Sans SC font
- ImageMagick 7 with the `magick` command

## Evaluation

```bash
lmms-eval --model <model> --tasks mdpbench --batch_size 1
```

The task prompt, prediction parser, element matching, and metric aggregation
are ported from the reference MDPBench implementation. Task-specific source
and third-party attribution is documented in `THIRD_PARTY_NOTICES.md`.

For chat-model backends, the task sends one user message containing the image
followed by the benchmark prompt. It does not insert a system message. The
legacy `doc_to_visual` and `doc_to_text` task interfaces remain available for
simple-model backends.

## References

- [MDPBench reference implementation](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/MDPBench)
- [CDM reference implementation](https://github.com/opendatalab/UniMERNet/tree/main/cdm)

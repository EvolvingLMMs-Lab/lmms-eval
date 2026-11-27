# ARC-AGI-2 (Image)

ARC-AGI-2 visual reasoning benchmark using community-rendered image dataset.

- Original: https://github.com/arcprize/ARC-AGI-2
- Dataset: https://huggingface.co/datasets/vincentkoc/arc-agi-2
- Note: Uses community image conversion (not official ARC Prize packaging)

## Usage
```bash
python -m lmms_eval --tasks arc_agi_2 --model <model_name> --batch_size 1
```

## Scoring
`arc_agi_2_acc` is 1 when the last integer grid in the response, inside or outside a code fence, exactly matches the expected output grid. Like the official ARC Prize harness, the last grid is used because a response often quotes the test input before its answer.

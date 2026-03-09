# VisRes-Bench

[VisRes-Bench](https://huggingface.co/datasets/tiiuae/visres_bench) is a visual reasoning benchmark with tasks at three difficulty levels. This folder defines lmms-eval tasks for all dataset configs.

**Dataset:** `tiiuae/visres_bench` (Hugging Face). A valid Hugging Face token may be required; set `HUGGINGFACE_HUB_TOKEN` or run `huggingface-cli login` before evaluation.

---

## Running the tasks

Use the `--tasks` argument with one of the group names or a single task name. Example with `accelerate launch`:

```bash
# From the lmms-eval repo root
accelerate launch --num_processes=1 -m lmms_eval \
    --model <your_model> \
    --model_args <your_args> \
    --tasks <TASK_OR_GROUP> \
    --batch_size 1
```

### Run all tasks (27 tasks)

```bash
--tasks visres_bench
```

Includes every config: all level-1 (including random_sampling), all level-2, and all level-3 tasks.

---

### Run Level 1 only (8 tasks, no random_sampling)

```bash
--tasks visres_bench_level_1
```

Tasks: `visres_bench_level_1_global_occlusion_50`, `visres_bench_level_1_global_occlusion_70`, `visres_bench_level_1_global_occlusion_80`, `visres_bench_level_1_edges`, `visres_bench_level_1_brightness`, `visres_bench_level_1_blur`, `visres_bench_level_1_rotation`, `visres_bench_level_1_location`.

---

### Run Level 2 only (12 tasks)

```bash
--tasks visres_bench_level_2
```

Tasks: `visres_bench_level_2_uniform_count`, `visres_bench_level_2_count_progression`, `visres_bench_level_2_uniform_orientation`, `visres_bench_level_2_count_2_same_1_diff`, `visres_bench_level_2_orientation_2same_1diff`, `visres_bench_level_2_uniform_color`, `visres_bench_level_2_count_arithmetic`, `visres_bench_level_2_count_minmax`, `visres_bench_level_2_orientation_3_diff`, `visres_bench_level_2_color_2same_1diff`, `visres_bench_level_2_color_3_diff`, `visres_bench_level_2_count_3_diff`.

---

### Run Level 3 only (5 tasks)

```bash
--tasks visres_bench_level_3
```

Tasks: `visres_bench_level_3_spiral_color_orientation`, `visres_bench_level_3_coupled_color_count`, `visres_bench_level_3_independent_color_object_rientation`, `visres_bench_level_3_coupled_color_orientation`, `visres_bench_level_3_Independent_count_object_color`.

---

## Single task

To run one config only, use the full task name, e.g.:

```bash
--tasks visres_bench_level_1_global_occlusion_50
```

---

## Question type (guided vs generic)

The default prompt uses the **guided_question** column. To use **generic_question** instead, pass the format that selects it (e.g. `--format generic` if your runner supports it). The default template defines:

- `default`: `question_column: guided_question`
- `generic`: `question_column: generic_question`

---

## Summary

| Group                 | Description              | # tasks |
|-----------------------|--------------------------|--------:|
| `visres_bench`        | All configs              | 27      |
| `visres_bench_level_1`| Level 1, no random_sampling | 8   |
| `visres_bench_level_2`| Level 2 only             | 12      |
| `visres_bench_level_3`| Level 3 only             | 5       |

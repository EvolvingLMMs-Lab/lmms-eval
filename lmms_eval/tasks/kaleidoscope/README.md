# Kaleidoscope

**In-language exams for massively multilingual vision evaluation.**

| | |
|---|---|
| Paper | [arXiv:2504.07072](https://arxiv.org/abs/2504.07072) (ICLR 2026) |
| Reference code | https://github.com/israfelsr/kaleidoscope |
| Dataset | [`CohereLabs/kaleidoscope`](https://huggingface.co/datasets/CohereLabs/kaleidoscope) |

Kaleidoscope is a multiple-choice exam benchmark built by an open-science
collaboration rather than by translating an English dataset: every question is
sourced from a real exam written in the language it is asked in. It covers
**20,911 questions**, **18 languages** and **14 subjects**; **11,457 questions
(54.8%)** carry an image that is needed to answer them.

Languages: Arabic, Bengali, Croatian, Dutch, English, French, German, Hindi,
Hungarian, Lithuanian, Nepali, Persian, Portuguese, Russian, Serbian, Spanish,
Telugu, Ukrainian.

## Tasks

| Task | Questions | Prompt regime | Paper setting |
|---|---|---|---|
| `kaleidoscope` | 20,911 | direct | group: multimodal + text-only |
| `kaleidoscope_multimodal` | 11,457 | direct | headline multimodal results (Table 2, Fig. 4) |
| `kaleidoscope_text_only` | 9,454 | direct | text-only column of Fig. 3a |
| `kaleidoscope_direct` | 20,911 | direct | all questions in one task |
| `kaleidoscope_multimodal_cot` | 11,457 | zero-shot CoT | closed-model multimodal setting |
| `kaleidoscope_cot` | 20,911 | zero-shot CoT | closed-model setting, all questions |

Both prompting regimes from the paper are implemented:

- **`direct`** — an English system message asking for `{"choice": "A"}` JSON,
  plus in-language `Question:` / `Options:` keywords and an `ANSWER:` cue. The
  paper uses this for open-weight models, which could not reliably follow the
  CoT format at ≤32B.
- **`cot`** — the zero-shot chain-of-thought system message, translated into all
  18 languages, asking the model to think step by step and close with
  `<ANSWER> X </ANSWER>`. The paper uses this for the closed models.

## Metrics

Matching `get_score.py` in the reference repo, three numbers are reported:

| Metric | Definition |
|---|---|
| `kaleidoscope_acc` | Accuracy over **all** samples; unparseable answers count as wrong. Macro-averaged over the 18 languages (equal weight per language). |
| `kaleidoscope_valid_acc` | Accuracy over samples whose answer could be extracted. Also macro-averaged over languages. |
| `kaleidoscope_format_error` | Share of samples (micro, over all) whose answer could not be extracted. Lower is better. |

The aggregation logs per-language, per-subject and — for multimodal tasks —
per-image-type breakdowns, so the tables in the paper's appendix can be
reproduced from one run.

## Usage

```bash
python -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=Qwen/Qwen2-VL-2B-Instruct \
    --tasks kaleidoscope_multimodal \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
```

### Images

The Hub release stores question metadata in parquet and the pixels in a
companion `data.zip` (~1 GB); the `image` column holds relative paths such as
`data/GATE_2022_Multimodal/images/xl_question_11.png`. Three resolution modes
are supported, all handled lazily inside `doc_to_visual`:

| Mode | How |
|---|---|
| Archive via HF cache (default) | Downloads `data.zip` once through `huggingface_hub` and reads members in place — no extraction, no second copy on disk. |
| Pre-extracted directory | `KALEIDOSCOPE_DATA_ROOT=/path/to/final_data` — the directory that directly contains `data/`. |
| Range-streamed archive | `KALEIDOSCOPE_STREAM_ZIP=1` — reads individual members over HTTP range requests without downloading the whole archive. Best for `--limit`ed smoke tests, too slow for a full run. |

### Environment overrides

| Variable | Default | Effect |
|---|---|---|
| `KALEIDOSCOPE_DATA_ROOT` | unset | Read images from a pre-extracted directory. |
| `KALEIDOSCOPE_STREAM_ZIP` | `0` | Stream archive members instead of downloading. |
| `KALEIDOSCOPE_IMAGE_SIZE` | `512` | Square edge length images are resized to; `0` feeds native resolution. |
| `KALEIDOSCOPE_LENIENT_EXTRACTION` | `0` | Fall back to the shared loose MCQ extractor when strict parsing fails. |

## Fidelity notes

Deliberate deviations from the reference implementation, all of them
side-effects of running inside lmms-eval rather than the paper's own harness:

1. **Inference backend.** The paper runs the Qwen models through vLLM; lmms-eval
   drives them through the HF `transformers` wrappers. Sampled decoding on a
   different backend produces different token sequences, so per-run numbers move
   even with identical prompts.
2. **System-message typo dropped.** The reference `SYS_MESSAGE` contains
   `\n\ONLY` — a Python escape-sequence slip that puts a literal backslash in
   front of `ONLY`, visible in the `prompt_used` field of its published outputs.
   The paper's appendix shows the intended text, which is what ships here.
3. **Sampling is non-deterministic by default.** `temperature=0.7`, `top_p=0.9`,
   `do_sample=true`, `max_new_tokens=1024` reproduce the reference sampling
   parameters. Add `--gen_kwargs temperature=0,do_sample=False` for repeatable
   runs.
4. **Answer extraction is marginally more forgiving.** `extract_choice` agrees
   with the reference `format_answer.py` on 199/200 of the published sample
   outputs. The single difference: the reference regex is line-bound and misses
   `{"choice": "C", "reasoning": "..."}` spread over several lines, which is
   counted here as a valid answer rather than a format error.
5. **Image options are shown as images.** Exactly 2 of the 20,911 questions have
   options that are themselves PNG paths. The reference open-model path renders
   those as the literal filename; here they are interleaved as images, matching
   the reference's own closed-model path.
6. **Model-side system prompt.** Chat wrappers (`is_simple = False`, the default
   resolution for e.g. `qwen2_5_vl`) receive the benchmark system message as a
   real `system` message through `doc_to_messages`. Legacy simple wrappers
   substitute their own system prompt, so for those the message is folded into
   the user turn by `doc_to_text` instead.
7. **`level`, `country` and few-shot splits** are carried in the records for
   breakdowns but no few-shot task is provided; the paper's headline results are
   zero-shot.

## Citation

```bibtex
@misc{salazar2025kaleidoscopeinlanguageexamsmassively,
      title={Kaleidoscope: In-language Exams for Massively Multilingual Vision Evaluation},
      author={Israfel Salazar and Manuel Fernández Burda and Shayekh Bin Islam and Arshia Soltani Moakhar and Shivalika Singh and Fabian Farestam and Angelika Romanou and Danylo Boiko and Dipika Khullar and Mike Zhang and Dominik Krzemiński and Jekaterina Novikova and Luísa Shimabucoro and Joseph Marvin Imperial and Rishabh Maheshwary and Sharad Duwal and Alfonso Amayuelas and Swati Rajwal and Jebish Purbey and Ahmed Ruby and Nicholas Popovič and Marek Suppa and Azmine Toushik Wasi and Ram Mohan Rao Kadiyala and Olga Tsymboi and Maksim Kostritsya and Bardia Soltani Moakhar and Gabriel da Costa Merlin and Otávio Ferracioli Coletti and Maral Jabbari Shiviari and MohammadAmin farahani fard and Silvia Fernandez and María Grandury and Dmitry Abulkhanov and Drishti Sharma and Andre Guarnier De Mitri and Leticia Bossatto Marchezi and Johan Obando-Ceron and Nazar Kohut and Beyza Ermis and Desmond Elliott and Enzo Ferrante and Sara Hooker and Marzieh Fadaee},
      year={2025},
      eprint={2504.07072},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.07072},
}
```

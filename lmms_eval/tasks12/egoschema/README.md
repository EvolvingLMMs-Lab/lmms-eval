# EgoSchema

## Task Description

<a href="https://github.com/egoschema/EgoSchema">EgoSchema</a>  is a diagnostic benchmark for very long-form video language understanding. The task format for EgoSchema is Multi-choice Question Answering.

- Questions: For each MCQ in the dataset, we provide a post prompt:`\nAnswer with the option's letter from the given choices directly.` 

- Answers: As required by the official website, we match the generated option letter into index, i.e., `A: 0, B: 1, C: 2, D: 3, E: 4.` Many models like LLaVA can follow the instructions well and generate only the option letter. However, in case model may generate redundant information (e.g., the entire option string, the option sentence, etc.), we also parse these outputs based on some pre-defined rule-based matching.

## Evaluation

### Full set: Submission

EgoSchema is intended for a 0-shot evaluation benchmark, hence the entire correct answer file will not be make public. 

`lmms-eval` will automatically generate a submission file `inference_results_egoschema_{taskname}_{now_date_time}.json` under `logs/`. To evaluate on the entire benchmark,  please submit the generated submission file using CURL:

`curl -X POST -H "Content-Type: application/json" -d @<path_to_json_file> https://validation-server.onrender.com/api/upload/`

### Subset: Direct Scoring

<a href="https://github.com/egoschema/EgoSchema">EgoSchema</a> also release the correct answers to only 500 of the EgoSchema questions provided in the subset_answers.json file intended for offline experimentation and performance tracking. Hence,`lmms-eval` will automatically generate the score for subset.

# Tasks

- `egoschema`: Standard MCQA for Full set. 
- `egoschema_mc_ppl`: MCQA Perplexity task format for Full set.
- `egoschema_subset`: Standard MCQA for Subset. 
- `egoschema_subset_mcppl`: MCQA Perplexity task format for Subset.
  
## Citation

```bibtex
@inproceedings{NEURIPS2023_90ce332a,
 author = {Mangalam, Karttikeya and Akshulakov, Raiymbek and Malik, Jitendra},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {46212--46244},
 publisher = {Curran Associates, Inc.},
 title = {EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/90ce332aff156b910b002ce4e6880dec-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}
```
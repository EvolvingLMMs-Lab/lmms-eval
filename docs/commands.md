# User Guide
This document details the interface exposed by `lmms_eval` and provides details on what flags are available to users.

## Command-line Interface


Equivalently, running the library can be done via the `lmms_eval` entrypoint at the command line.

This mode supports a number of command-line arguments, the details of which can be also be seen via running with `-h` or `--help`:

* `--model` : Selects which model type or provider is evaluated. Must be a mdoels registered under lmms_eval/models. For example, `--model qwen_vl` or `--model llava`.

* `--model_args` : Controls parameters passed to the model constructor. Accepts a string containing comma-separated keyword arguments to the model class of the format `"arg1=val1,arg2=val2,..."`, such as, for example `--model_args pretrained=liuhaotian/llava-v1.5-7b,batch_size=1`. For a full list of what keyword arguments, see the initialization of the corresponding model class in `lmms_eval/models/`.

* `--tasks` : Determines which tasks or task groups are evaluated. Accepts a comma-separated list of task names or task group names. Must be solely comprised of valid tasks/groups. You can use `--tasks list` to see all the available tasks. If you add your own tasks but not shown on the list, you can try to set `--verbosity=DEBUG` to view the error message. You can also use `--tasks list_with_num` to check every tasks and the number of question each task contains. However, `list_with_num` will download all the available datasets and may require lots of memory and time.

* `--batch_size` : Sets the batch size used for evaluation. Can be a positive integer or `"auto"` to automatically select the largest batch size that will fit in memory, speeding up evaluation. One can pass `--batch_size auto:N` to re-select the maximum batch size `N` times during evaluation. This can help accelerate evaluation further, since `lm-eval` sorts documents in descending order of context length.

* `--output_path` : A string of the form `dir/file.jsonl` or `dir/`. Provides a path where high-level results will be saved, either into the file named or into the directory named. If `--log_samples` is passed as well, then per-document outputs and metrics will be saved into the directory as well.

* `--log_samples` : If this flag is passed, then the model's outputs, and the text fed into the model, will be saved at per-document granularity. Must be used with `--output_path`.

* `--limit` : Accepts an integer, or a float between 0.0 and 1.0 . If passed, will limit the number of documents to evaluate to the first X documents (if an integer) per task or first X% of documents per task. Useful for debugging, especially on costly API models.


import collections
import datetime
import fnmatch
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import gc
from itertools import islice

import numpy as np
import pytz
import torch
import transformers
from jinja2 import BaseLoader, Environment, StrictUndefined
from loguru import logger as eval_logger

SPACING = " " * 47
HIGHER_IS_BETTER_SYMBOLS = {
    True: "↑",
    False: "↓",
}


def is_json(string):
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def escaped_split(text, sep_char, maxsplit=-1):
    """Split text into a list on occurrences of the given separation
    character `sep_char`. The separation character may be escaped by a
    backslash to avoid splitting at that location.

    The separation character must be a string of size 1.

    If `maxsplit` is given, at most `maxsplit` splits are done (thus,
    the list will have at most `maxsplit + 1` elements). If `maxsplit`
    is not specified or less than 0, then there is no limit on the
    number of splits (all possible splits are made).
    """
    assert len(sep_char) == 1, "separation string must be a single character for escaped splitting"

    if maxsplit == 0:
        return text
    maxsplit = max(0, maxsplit)

    return re.split(r"(?<!\\)" + sep_char, text, maxsplit)


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def is_multimodal_content(value: Any) -> bool:
    """
    Check if a value is multimodal content (image, audio, video) that should
    not be serialized to log files.

    Returns True for:
    - PIL.Image objects
    - numpy arrays (typically image/audio data)
    - bytes (binary data)
    - torch tensors
    - dicts with 'array' key (HuggingFace audio format)
    - dicts with 'bytes' key (HuggingFace image format)
    """
    if isinstance(value, (bytes, bytearray, np.ndarray, torch.Tensor)):
        return True
    if isinstance(value, dict):
        if "array" in value or "bytes" in value:
            return True
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            return True
    except ImportError:
        pass
    return False


def sanitize_list(sub):
    """
    Takes possible nested list and recursively converts all inner component to strings
    """
    if isinstance(sub, list):
        return [sanitize_list(item) for item in sub]
    if isinstance(sub, tuple):
        return tuple(sanitize_list(item) for item in sub)
    else:
        return str(sub)


def _smart_comma_split(args_string):
    """
    Splits a string by comma, but not if inside quotes or braces.

    This is useful for parsing argument strings that may contain JSON or
    other structured values with nested commas.

    Args:
        args_string: The string to split

    Returns:
        List of split arguments
    """
    arg_list = []
    current_arg = []
    depth = 0  # Track nesting depth of braces/brackets
    in_quotes = False
    quote_char = None

    for i, char in enumerate(args_string):
        if char in ('"', "'") and (i == 0 or args_string[i - 1] != "\\"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        elif not in_quotes:
            if char in ("{", "["):
                depth += 1
            elif char in ("}", "]"):
                depth -= 1
            elif char == "," and depth == 0:
                # This is a separator comma, not inside JSON
                arg = "".join(current_arg).strip()
                if arg:
                    arg_list.append(arg)
                current_arg = []
                continue

        current_arg.append(char)

    # Don't forget the last argument
    arg = "".join(current_arg).strip()
    if arg:
        arg_list.append(arg)

    return arg_list


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    Handles JSON values with nested commas, e.g., arg='{"key1":val1,"key2":val2}'
    """
    args_string = args_string.strip()
    if not args_string:
        return {}

    # Smart split: split by comma, but not if inside quotes or braces
    arg_list = _smart_comma_split(args_string)

    args_dict = {k: handle_arg_string(v) for k, v in [arg.split("=", 1) for arg in arg_list]}
    return args_dict


def join_iters(iters):
    for iter in iters:
        yield from iter


def chunks(iter, n: int = 0, fn=None):
    """
    Divides an iterable into chunks of specified size or based on a given function.
    Useful for batching

    Parameters:
    - iter: The input iterable to be divided into chunks.
    - n: An integer representing the size of each chunk. Default is 0.
    - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

    Returns:
    An iterator that yields chunks of the input iterable.

    Example usage:
    ```
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for chunk in chunks(data, 3):
        print(chunk)
    ```
    Output:
    ```
    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    [10]
    ```
    """
    arr = []
    for i, x in enumerate(iter):
        arr.append(x)
        if len(arr) == (fn(i, iter) if fn else n):
            yield arr
            arr = []

    if arr:
        yield arr


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


class MultiChoice:
    def __init__(self, choices) -> None:
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values) -> bool:
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                eval_logger.info(f"Available tasks to choose:")
                for choice in self.choices:
                    eval_logger.info(f"  - {choice}")
                raise ValueError("'{}' is not in task list".format(value))
        return True

    def __iter__(self) -> Iterator:
        for choice in self.choices:
            yield choice


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    if type(patterns) == str:
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        try:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        except Exception as e:
            eval_logger.error(f"Error matching pattern {pattern}: {e}")
    return sorted(list(task_names))


def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string


def get_file_task_name(filename: str) -> str:
    """
    Given the sample results filenames, extracts and returns the task name.
    """
    return filename[filename.find("_") + 1 : filename.rfind("_")]


def get_file_datetime(filename: str) -> str:
    """
    Given the results and sample results filenames, extracts and returns the datetime.
    """
    return filename[filename.rfind("_") + 1 :].replace(".jsonl", "")


def sanitize_model_name(model_name: str, full_path: bool = False) -> str:
    """
    Given the model name, returns a sanitized version of it.
    """
    if full_path:
        return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_name)
    else:
        parts = model_name.split("/")
        last_two = "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]  # accommondate for models that are in Hugging Face Hub format like lmms-lab/llava-onevision-qwen2-0.5b
        return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", last_two)


def sanitize_task_name(task_name: str) -> str:
    """
    Given the task name, returns a sanitized version of it.
    """
    return re.sub(r"\W", "_", task_name)


def get_latest_filename(filenames: List[str]) -> str:
    """
    Given a list of filenames, returns the filename with the latest datetime.
    """
    return max(filenames, key=lambda f: get_file_datetime(f))


def get_results_filenames(filenames: List[str]) -> List[str]:
    """
    Extracts filenames that correspond to aggregated results.
    """
    return [f for f in filenames if "results" in f and ".json" in f]


def get_sample_results_filenames(filenames: List[str]) -> List[str]:
    """
    Extracts filenames that correspond to sample results.
    """
    return [f for f in filenames if "/samples_" in f and ".json" in f]


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LMM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def make_disjoint_window(pair):
    """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Provides a proper json encoding for the loggers and trackers json dumps.
    Notably manages the json encoding of dataclasses.
    """

    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


class Reorderer:
    def __init__(self, arr: List[Any], fn: Callable) -> None:
        """Reorder an array according to some function

        Args:
            arr (List[Any]): The initial array
            fn (Callable[[Any], Any]): A function to determine the priority of elements
        """
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        # arr = [([y[0] for y in x], x[0][1]) for x in arr]
        # TODO: overhaul reorderer. It currently grouped requests by content but we don't want this
        arr = [([y[0]], x[0][1]) for x in arr for y in x]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr

    def get_reordered(self):
        """Gets the reordered array

        Returns:
            List[Any]: The reordered array
        """
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        """Restores the original order of a new array based on the old array's order

        Args:
            newarr (List[Any]): The array to be restored

        Returns:
            List[Any]: The array restored to the original order
        """
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res


class Grouper:
    """
    takes an array `arr` and function `fn` and returns a dictionary
    with keys fn(ob) for each ob in `arr` and with values `self.arr[key]` a list of all
    objects in `arr` satisfying `key == fn(ob)`.
    """

    def __init__(self, arr, fn) -> None:
        # self.orig_arr = arr
        self.size = len(arr)
        arr = list(enumerate(arr))

        def group_return_dict(arr, fn):
            res = collections.defaultdict(list)

            for ob in arr:
                res[fn(ob)].append(ob)
            return res

        arr = group_return_dict(arr, lambda x: fn(x[1]))

        # self.arr has format Dict[Tuple[int, <entry from orig. arr>]]
        self.arr = arr
        self._grouped = None

    def get_grouped(self):
        # return the contents but not indices for our grouped dict.
        if self._grouped:
            return self._grouped
        grouped = {}
        for key in self.arr.keys():
            # drop the index from each element of self.arr
            grouped[key] = [y[1] for y in self.arr[key]]
        self._grouped = grouped
        return grouped

    def get_original(self, grouped_dict):
        # take in a grouped dictionary with e.g. results for each key listed
        # in the same order as the instances in `self.arr`, and
        # return the results in the same (single list) order as `self.orig_arr`.
        res = [None] * self.size
        cov = [False] * self.size
        # orig = [None] * self.size

        assert grouped_dict.keys() == self.arr.keys()

        for key in grouped_dict.keys():
            for (ind, _), v in zip(self.arr[key], grouped_dict[key]):
                res[ind] = v
                cov[ind] = True
                # orig[ind] = _

        assert all(cov)
        # assert orig == self.orig_arr

        return res


def make_table(result_dict, column: str = "results", sort_results: bool = False):
    """Generate table of results."""
    from pytablewriter import LatexTableWriter, MarkdownTableWriter

    if column == "results":
        column_name = "Tasks"
    elif column == "groups":
        column_name = "Groups"

    all_headers = [
        column_name,
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = all_headers
    latex_writer.headers = all_headers

    values = []

    keys = result_dict[column].keys()
    if sort_results:
        # sort entries alphabetically by task or group name.
        # NOTE: we default here to false, because order matters for multi-level table printing a la mmlu.
        # sorting here would mess that up
        keys = sorted(keys)
    for k in keys:
        dic = result_dict[column][k]
        version = result_dict["versions"].get(k, "    N/A")
        n = str(result_dict.get("n-shot", " ").get(k, " "))
        higher_is_better = result_dict.get("higher_is_better", {}).get(k, {})

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        metric_items = sorted(metric_items)

        for (mf), v in metric_items:
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), "")

            v = "%.4f" % v if isinstance(v, float) else v
            if v == "" or v is None:
                v = "N/A"

            if m + "_stderr" + "," + f in dic:
                # if dic[m + "_stderr" + "," + f] != []:
                se = dic[m + "_stderr" + "," + f]
                try:
                    se = "   N/A" if se == "N/A" or se == [] else "%.4f" % se
                except:
                    se = "N/A"
                if v != []:
                    values.append([k, version, f, n, m, hib, v, "±", se])
            else:
                values.append([k, version, f, n, m, hib, v, "", ""])
            # k = ""
            # version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


def positional_deprecated(fn):
    """
    A decorator to nudge users into passing only keyword args (`kwargs`) to the
    wrapped function, `fn`.
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(f"WARNING: using {fn.__name__} with positional arguments is " "deprecated and will be disallowed in a future version of " "lmms-evaluation-harness!")
        return fn(*args, **kwargs)

    return _wrapper


@positional_deprecated
def find_test_root(start_path: pathlib.Path) -> pathlib.Path:
    """
    Search upward in the directory tree to a maximum of three layers
    to find and return the package root (containing the 'tests' folder)
    """
    cur_path = start_path.resolve()
    max_layers = 3
    for _ in range(max_layers):
        if (cur_path / "tests" / "test_version_stable.py").exists():
            return cur_path
        else:
            cur_path = cur_path.parent.resolve()
    raise FileNotFoundError(f"Unable to find package root within {max_layers} upwards" + f"of {start_path}")


@positional_deprecated
def run_task_tests(task_list: List[str]):
    """
    Find the package root and run the tests for the given tasks
    """
    import pytest

    package_root = find_test_root(start_path=pathlib.Path(__file__))
    task_string = " or ".join(task_list)
    args = [
        f"{package_root}/tests/test_version_stable.py",
        f"--rootdir={package_root}",
        "-k",
        f"{task_string}",
    ]
    sys.path.append(str(package_root))
    pytest_return_val = pytest.main(args)
    if pytest_return_val:
        raise ValueError(f"Not all tests for the specified tasks ({task_list}) ran successfully! Error code: {pytest_return_val}")


def get_git_commit_hash():
    """
    Gets the git commit hash of your current repo (if it exists).
    Source: https://github.com/EleutherAI/gpt-neox/blob/b608043be541602170bfcfb8ec9bf85e8a0799e0/megatron/neox_arguments/neox_args.py#L42
    """
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
        git_hash = git_hash.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # FileNotFoundError occurs when git not installed on system
        git_hash = None
    return git_hash


def get_datetime_str(timezone="Asia/Singapore"):
    """
    Gets the current datetime in UTC+8 timezone as a string.
    """
    # Default: UTC+8 timezone
    tz = pytz.timezone(timezone)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    local_time = utc_now.astimezone(tz)
    return local_time.strftime("%Y%m%d_%H%M%S")


def sanitize_long_string(s, max_length=40):
    if len(s) > max_length:
        return s[: max_length // 2] + "..." + s[-max_length // 2 :]
    return s


def ignore_constructor(loader, node):
    return node


def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)

    *module_name, function_name = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)

    # 1) Try relative file import (original behavior)
    module_path = os.path.normpath(os.path.join(yaml_path, "{}.py".format(module_name)))
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name)
        return function

    # 2) Fallback to absolute module import (enables cross-task utils like
    #    lmms_eval.tasks.librispeech.utils.librispeech_doc_to_audio)
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function
    except Exception as ex:
        # Re-raise with context to aid debugging
        raise ImportError(f"Failed to import function '{function_name}' from module '{module_name}'. " f"Tried relative path '{module_path}' and absolute import.") from ex


def load_yaml_config(yaml_path=None, yaml_config=None, yaml_dir=None, mode="full"):
    if mode == "simple":
        constructor_fn = ignore_constructor
    elif mode == "full":
        constructor_fn = import_function

    # Add the import_function constructor to the YAML loader
    yaml.add_constructor("!function", constructor_fn)
    if yaml_config is None:
        with open(yaml_path, "rb") as file:
            yaml_config = yaml.full_load(file)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    assert yaml_dir is not None
    assert yaml_config is not None

    if "include" in yaml_config:
        include_path = yaml_config["include"]
        del yaml_config["include"]

        if isinstance(include_path, str):
            include_path = [include_path]

        # Load from the last one first
        include_path.reverse()
        final_yaml_config = {}
        for path in include_path:
            # Assumes that path is a full path.
            # If not found, assume the included yaml
            # is in the same dir as the original yaml
            if not os.path.isfile(path):
                path = os.path.join(yaml_dir, path)

            try:
                included_yaml_config = load_yaml_config(yaml_path=path, mode=mode)
                final_yaml_config.update(included_yaml_config)
            except Exception as ex:
                # If failed to load, ignore
                raise ex

        final_yaml_config.update(yaml_config)
        return final_yaml_config
    return yaml_config


def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)


env = Environment(loader=BaseLoader, undefined=StrictUndefined)
env.filters["regex_replace"] = regex_replace


def apply_template(template: str, doc: dict) -> str:
    rtemplate = env.from_string(template)
    return rtemplate.render(**doc)


def create_iterator(raw_iterator, rank, world_size, limit=None):
    """
    Method for creating a (potentially) sliced and limited
    iterator from a raw document iterator. Used for splitting data
    among ranks in multigpu setting or only pulling a sample of documents
    """
    return islice(raw_iterator, rank, limit, world_size)


def pad_and_concat(
    max_length: int,
    tensors: List[torch.Tensor],
    padding_side: Literal["right", "left"] = "right",
):
    """
    Method for padding a list of tensors given the maximum tensor
    length in the batch. Used for batching inputs and continuations in
    seq2seq models.
    """
    assert padding_side == "left" or padding_side == "right", f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"

    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                tensors[i] = torch.cat(
                    [
                        tensor,  # [seq]
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
            else:
                # left-pad
                tensors[i] = torch.cat(
                    [
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                        tensor,  # [seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
        else:
            tensors[i] = tensor.unsqueeze(0)

    return torch.cat(tensors, dim=0)


def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


# Multi-token stopping criteria
class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[MultiTokenEOSCriteria(sequence, tokenizer, initial_decoder_input_length, batch_size) for sequence in stop_sequences],
        ]
    )


# from more_itertools
def divide(iterable, n) -> List[Iterator]:
    """Divide the elements from *iterable* into *n* parts, maintaining
    order.

        >>> group_1, group_2 = divide(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 2, 3]
        >>> list(group_2)
        [4, 5, 6]

    If the length of *iterable* is not evenly divisible by *n*, then the
    length of the returned iterables will not be identical:

        >>> children = divide(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 2, 3], [4, 5], [6, 7]]

    If the length of the iterable is smaller than n, then the last returned
    iterables will be empty:

        >>> children = divide(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]

    This function will exhaust the iterable before returning and may require
    significant storage. If order is not important, see :func:`distribute`,
    which does not first pull the iterable into memory.

    """
    if n < 1:
        raise ValueError("n must be at least 1")

    try:
        iterable[:0]
    except TypeError:
        seq = tuple(iterable)
    else:
        seq = iterable

    q, r = divmod(len(seq), n)

    ret = []
    stop = 0
    for i in range(1, n + 1):
        start = stop
        stop += q + 1 if i <= r else q
        ret.append(iter(seq[start:stop]))

    return ret


class Collator:
    """
    A class for reordering and batching elements of an array.

    This class allows for sorting an array based on a provided sorting function, grouping elements based on a grouping function, and generating batches from the sorted and grouped data.
    """

    def __init__(
        self,
        arr: List,
        sort_fn: Callable,
        group_fn: Callable = lambda x: x[1],
        grouping: bool = False,
    ) -> None:
        self.grouping = grouping
        self.fn = sort_fn
        self.group_fn = lambda x: group_fn(x[1])  # first index are enumerated indices
        self.reorder_indices: List = []
        self.size = len(arr)
        self.arr_with_indices: Iterable[Any] = tuple(enumerate(arr))  # [indices, (arr)]
        if self.grouping is True:
            self.group_by_index()

    def group_by_index(self) -> None:
        self.arr_with_indices = self.group(self.arr_with_indices, fn=self.group_fn, values=False)

    def get_batched(self, n: int = 1, batch_fn: Optional[Callable] = None) -> Iterator:
        """
        Generates and yields batches from the reordered array.

        Parameters:
        - n (int): The size of each batch. Defaults to 1.
        - batch_fn (Optional[Callable[[int, Iterable], int]]): A function to determine the size of each batch. Defaults to None.

        Yields:
        Iterator: An iterator over batches of reordered elements.
        """
        if self.grouping:
            for (
                key,
                values,
            ) in self.arr_with_indices.items():  # type: ignore
                values = self._reorder(values)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch
        else:
            values = self._reorder(self.arr_with_indices)  # type: ignore
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def _reorder(self, arr: Union[List, Tuple[Tuple[int, Any], ...]]) -> List:
        """
        Reorders the elements in the array based on the sorting function.

        Parameters:
        - arr (Union[List, Tuple[Tuple[int, Any], ...]]): The array or iterable to be reordered.

        Yields:
        List: Yields reordered elements one by one.
        """
        arr = sorted(arr, key=lambda x: self.fn(x[1]))
        self.reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def get_original(self, newarr: List) -> List:
        """
        Restores the original order of elements from the reordered list.

        Parameters:
        - newarr (List): The reordered array.

        Returns:
        List: The array with elements restored to their original order.
        """
        res = [None] * self.size
        cov = [False] * self.size

        for ind, v in zip(self.reorder_indices, newarr):
            res[ind] = v
            cov[ind] = True

        assert all(cov)

        return res

    def __len__(self):
        return self.size

    @staticmethod
    def group(arr: Iterable, fn: Callable, values: bool = False) -> Iterable:
        """
        Groups elements of an iterable based on a provided function.

        Parameters:
        - arr (Iterable): The iterable to be grouped.
        - fn (Callable): The function to determine the grouping.
        - values (bool): If True, returns the values of the group. Defaults to False.

        Returns:
        Iterable: An iterable of grouped elements.
        """
        res = collections.defaultdict(list)
        for ob in arr:
            try:
                hashable_dict = tuple(
                    (
                        key,
                        tuple(value) if isinstance(value, collections.abc.Iterable) else value,
                    )
                    for key, value in sorted(fn(ob).items())
                )
                res[hashable_dict].append(ob)
            except TypeError:
                res[fn(ob)].append(ob)
        if not values:
            return res
        return res.values()

    @staticmethod
    def get_chunks(_iter, n: int = 0, fn=None):
        """
        Divides an iterable into chunks of specified size or based on a given function.
        Useful for batching

        Parameters:
        - iter: The input iterable to be divided into chunks.
        - n: An integer representing the size of each chunk. Default is 0.
        - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

        Returns:
        An iterator that yields chunks of the input iterable.

        Example usage:
        ```
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for chunk in chunks(data, 3):
            print(chunk)
        ```
        Output:
        ```
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
        ```
        """
        arr = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            arr.append(x)
            if len(arr) == (fn(i, _iter) if fn else n):
                yield arr
                arr = []

        if arr:
            yield arr

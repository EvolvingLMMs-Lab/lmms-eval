import abc
import collections
import gc
import hashlib
import json
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger as eval_logger
from sqlitedict import SqliteDict
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance

T = TypeVar("T", bound="lmms")

LMMS_EVAL_HOME = os.path.expanduser(os.getenv("LMMS_EVAL_HOME", "~/.cache/lmms-eval"))
LMMS_EVAL_USE_CACHE = os.getenv("LMMS_EVAL_USE_CACHE", "False")


class lmms(abc.ABC):
    is_simple: bool = True

    def __init__(self) -> None:
        """Defines the interface that should be implemented by all lmms subclasses.
        lmmss are assumed to take image-text as input and yield strings as output
        (inputs/outputs should be tokenization-agnostic.)
        """
        # set rank and world size to a single process, by default.
        self._rank = 0
        self._world_size = 1
        self.cache_hook = CacheHook(None)
        self.task_dict = {}
        self.cache_dict = collections.defaultdict(dict)
        self.initialized_cache_dir = False

    def prepare_cache_dir(self):
        if self.initialized_cache_dir:
            return
        # initialize cache directory for this model instance
        resolved_name = self._resolve_model_name_for_cache()
        cache_hash = self.generate_cache_folder_hash_name(resolved_name)
        self._cache_dir = os.path.join(LMMS_EVAL_HOME, "eval_cache", cache_hash)
        eval_logger.info(f"Resolved model folder for cache: {self._cache_dir}")
        self.initialized_cache_dir = True

    def generate_cache_folder_hash_name(self, model_name: str):
        """
        Generate a cache hash for a model
        """
        task_dict_keys = list(self.task_dict.keys())
        class_name = type(self).__name__
        hash_string = "|".join(task_dict_keys)

        text_hash = unicodedata.normalize("NFC", hash_string)
        text_hash = text_hash.replace("\r\n", "\n").replace("\r", "\n")

        hash_string = hashlib.sha256(text_hash.encode("utf-8")).hexdigest()
        model_name = os.path.basename(model_name)
        folder_name = class_name + "_" + model_name + "_" + hash_string
        return folder_name

    def _resolve_model_name_for_cache(self) -> str:
        """
        Best-effort resolution of a human-readable model identifier for cache naming.
        Checks common attributes; falls back to class name.
        """
        for attr_name in ("model_name", "model_version", "model_id", "pretrained"):
            value = getattr(self, attr_name, None)
            if isinstance(value, str) and value:
                return value
        value = getattr(self, "model", None)
        if isinstance(value, str) and value:
            return value
        return type(self).__name__

    @property
    def get_model_cache_dir(self) -> str:
        """
        Property returning the initialized cache directory for this model instance.
        """
        return self._cache_dir

    def get_rank_and_world_size(self) -> Tuple[int, int]:
        """
        Get the rank and world size for the current process
        """
        # The rank and world size is a bit chaotic in current many ... many model implementations
        # So we use torch.distributed to get the rank and world size here instead of self.rank and self.world_size
        # fallback if not initialized
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return self.rank, self.world_size

    def ensure_model_cache_dir(self) -> str:
        """
        Ensure the cache directory for a given model exists, and return its path.
        """
        os.makedirs(self.get_model_cache_dir, exist_ok=True)
        return self.get_model_cache_dir

    def load_cache(self):
        if LMMS_EVAL_USE_CACHE == "True":
            self.prepare_cache_dir()
            self.cache_dict = self.load_jsonl_cache()
        else:
            self.cache_dict = collections.defaultdict(dict)

    def load_jsonl_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all .jsonl files in the model's cache directory.

        Returns a dict mapping filename (base name) -> list of records.
        Missing directory returns empty dict.
        """
        cache_dir = self.get_model_cache_dir
        if not os.path.isdir(cache_dir):
            return collections.defaultdict(dict)

        rank, world_size = self.get_rank_and_world_size()

        files = [f"{task_name}_rank{rank}_world_size{world_size}.jsonl" for task_name in self.task_dict.keys()]

        cache_data: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
        try:
            for task_name, fname in zip(self.task_dict.keys(), files):
                full_path = os.path.join(cache_dir, fname)
                records: Dict[str, Any] = collections.defaultdict(dict)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            try:
                                line = json.loads(line)
                                records[line["doc_id"]] = line["response"]
                            except (json.JSONDecodeError, KeyError):
                                eval_logger.warning(f"Skipping malformed JSONL line in {full_path}")
                except FileNotFoundError:
                    # If file disappears during read, skip
                    continue
                cache_data[task_name] = records.copy()
        except FileNotFoundError:
            # Directory disappeared between checks
            return collections.defaultdict(dict)
        except Exception as e:
            eval_logger.error(f"Error loading cache from {full_path}: {e}")
            return collections.defaultdict(dict)

        return cache_data

    def _extract_doc_id(self, request: Instance) -> Any:
        """
        TODO: Implement logic to extract `doc_id` from a request.
        This method should return a JSON-serializable identifier (e.g., int or str).
        """
        try:
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        except Exception as e:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.arguments
        return doc_id

    def _append_request_response_to_cache(
        self,
        request: Instance,
        response: str,
        task_name: str,
    ) -> str:
        """
        Append a single request/response record to a JSONL cache file under the
        model's cache directory. The record format is:
        {"doc_id": <doc_id>, "response": <response>}

        Returns the full path of the file written to.
        """
        cache_dir = self.ensure_model_cache_dir()

        rank, world_size = self.get_rank_and_world_size()

        base = f"{task_name}_rank{rank}_world_size{world_size}.jsonl"

        file_path = os.path.join(cache_dir, base)

        # Obtain doc_id via user-implemented logic
        doc_id = self._extract_doc_id(request)

        record = {"doc_id": doc_id, "response": response}
        self.cache_dict[task_name][doc_id] = record
        line = json.dumps(record, ensure_ascii=False)

        # Append in text mode with UTF-8 encoding
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        return file_path

    def add_request_response_to_cache(self, request: Instance, response: str):
        """
        Add a request and response to the cache
        """
        if LMMS_EVAL_USE_CACHE == "True":
            self._append_request_response_to_cache(request, response, request.task_name)

    def get_response_from_cache(self, requests: List[Instance]) -> Tuple[List[str], List[Instance]]:
        """
        Get the response from the cache
        """
        if LMMS_EVAL_USE_CACHE == "False":
            return [], requests
        not_cached_requests = []
        responses = []
        for request in requests:
            if request.doc_id not in self.cache_dict[request.task_name]:
                not_cached_requests.append(request)
            else:
                responses.append(self.cache_dict[request.task_name][request.doc_id])
        eval_logger.info(f"Loaded {len(responses)} responses from cache")
        eval_logger.info(f"Not cached {len(not_cached_requests)} requests")
        return responses, not_cached_requests

    @abc.abstractmethod
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LMM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LMM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
            'visual_list: list[dict]'
                Visual input to the model. Can be None.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        pass

    # TODO: Add an optional max length
    @abc.abstractmethod
    def generate_until(self, requests) -> List[str]:
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            generation_kwargs: dict
                Generation Kwargs
            'visual_list: list[dict]'
                Visual input to the model. Can be None.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        pass

    @abc.abstractmethod
    def generate_until_multi_round(self, requests) -> List[str]:
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            generation_kwargs: dict
                Generation Kwargs
            'visual_list: list[dict]'
                Visual input to the model. Can be None.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        pass

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict] = None) -> T:
        """
        Creates an instance of the LMM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LMM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    @property
    def rank(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._rank

    @property
    def world_size(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._world_size

    def set_cache_hook(self, cache_hook) -> None:
        self.cache_hook = cache_hook

    def clean(self):
        for attr_name in list(vars(self)):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, nn.Module):
                delattr(self, attr_name)
        gc.collect()
        torch.cuda.empty_cache()


### SQLite-based caching of LMM responses
def hash_args(attr, args):
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


class CacheHook:
    def __init__(self, cachinglm) -> None:
        if cachinglm is None:
            self.dbdict = None
            return

        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr, req, res) -> None:
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


class CachingLMM:
    def __init__(self, lm, cache_db) -> None:
        """LMM wrapper that returns cached results if they exist, and uses the underlying LMM if not.

        :param lm: LMM
            Underlying LMM
        :param cache_db: str
            Path to cache db
        """
        self.lm = lm
        self.cache_db = cache_db
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.dbdict = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr):
        lm_attr = getattr(self.lm, attr)
        if not callable(lm_attr):
            return lm_attr

        def fn(requests):
            res = []
            remaining_reqs = []
            warned = False
            # figure out which ones are cached and which ones are new
            eval_logger.info(f"Loading '{attr}' responses from cache '{self.cache_db}' where possible...")
            for req in tqdm(requests):
                hsh = hash_args(attr, req.args)
                if attr in ["generate_until", "generate_until_multi_round"] and req.args[1].get("do_sample", False):
                    # when we are doing non-greedy generation, don't use the cache
                    # (else every "randomly sampled" generation would be identical for repeats > 1).
                    if not warned:
                        eval_logger.warning(f"Arguments to lm.generate_until() '{req.args[1]}' include non-deterministic sampling. Caching will not be performed for such requests.")
                        warned = True
                    res.append(None)
                    remaining_reqs.append(req)
                elif hsh in self.dbdict:
                    ob = self.dbdict[hsh]

                    assert ob is not None

                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append(req)

            # actually run the LMM on the requests that do not have cached results
            rem_res = getattr(self.lm, attr)(remaining_reqs)

            # stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res):
                while res[resptr] is not None:
                    resptr += 1

                res[resptr] = r

                # caching
                hsh = hash_args(attr, req.args)
                self.dbdict[hsh] = r
            self.dbdict.commit()

            return res

        return fn

    def get_cache_hook(self):
        return CacheHook(self)

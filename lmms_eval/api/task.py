import abc
from dataclasses import dataclass, field, asdict

import itertools
import os
import re
import ast
import logging
import random
from glob import glob
import shutil
from tqdm import tqdm

import datasets

datasets.config.STREAMING_READ_MAX_RETRIES = 20  # default
datasets.config.STREAMING_READ_RETRY_INTERVAL = 5  # default

from datasets import Image, Sequence
import numpy as np
from PIL import ImageFile

from accelerate import Accelerator
from datasets import DownloadConfig
from typing import Union, List, Any
from collections.abc import Callable
from tenacity import retry, stop_after_attempt, wait_fixed, stop_after_delay
from huggingface_hub import snapshot_download

from lmms_eval import utils
from lmms_eval.api import samplers
from lmms_eval.api.instance import Instance

from lmms_eval.filters import build_filter_ensemble
from lmms_eval.api.registry import (
    get_aggregation,
    get_metric_aggregation,
    is_higher_better,
    DEFAULT_METRIC_REGISTRY,
    METRIC_REGISTRY,
    OUTPUT_TYPE_REGISTRY,
    AGGREGATION_REGISTRY,
)

ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "generate_until",
]


eval_logger = logging.getLogger("lmms-eval")

# HuggingfaceM4/NoCaps contains truncated image in test split
# Include this inside code block to avoid error
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class TaskConfig(dict):
    # task naming/registry
    task: str = None
    task_alias: str = None
    group: Union[str, list] = None
    group_alias: Union[str, list] = None
    # HF dataset options.
    # which dataset to use,
    # and what splits for what purpose
    dataset_path: str = None
    dataset_name: str = None
    dataset_kwargs: dict = None
    training_split: str = None
    validation_split: str = None
    test_split: str = None
    fewshot_split: str = None  # TODO: assert that this not None if num_fewshot > 0. (?) assert if this is same split as one evaling (?)
    # formatting / prompting options.
    # see docs/advanced_task_guide.md for more info
    process_docs: Callable = None
    doc_to_visual: Union[Callable, str] = None
    doc_to_text: Union[Callable, str] = None
    doc_to_target: Union[Callable, str] = None
    doc_to_choice: Union[Callable, str, dict, list] = None
    process_results: Union[Callable, str] = None
    use_prompt: str = None
    description: str = ""
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    fewshot_config: dict = None
    # runtime configuration options
    num_fewshot: int = None
    # scoring options
    metric_list: list = None
    output_type: str = "generate_until"
    generation_kwargs: dict = None
    repeats: int = 1
    filter_list: Union[str, list] = None
    should_decontaminate: bool = False
    doc_to_decontamination_query: str = None

    metadata: Union[str, list] = None  # by default, not used in the code. allows for users to pass arbitrary info to tasks

    model_specific_prompt_kwargs: dict = None
    model_specific_generation_kwargs: dict = None
    model_specific_target_kwargs: dict = None

    def __post_init__(self) -> None:
        if self.dataset_path and os.path.exists(os.path.dirname(self.dataset_path)):
            import inspect
            from importlib import import_module

            self.dataset_path = inspect.getfile(import_module(self.dataset_path))

        if self.generation_kwargs is not None:
            if self.output_type != "generate_until":
                eval_logger.warning(f"[{self.task}] passed `generation_kwargs`, but not using `output_type: generate_until`!")
                assert self.output_type != "generate_until"

            if "temperature" in self.generation_kwargs:
                self.generation_kwargs["temperature"] = float(self.generation_kwargs["temperature"])

            if "until" not in self.generation_kwargs:
                self.generation_kwargs["until"] = [self.fewshot_delimiter]
        else:
            if self.output_type == "generate_until":
                # ensure that we greedily generate in absence of explicit arguments otherwise
                self.generation_kwargs = {
                    "until": None if self.fewshot_delimiter is None else [self.fewshot_delimiter],
                    "do_sample": False,
                }

        # TODO: how to make TaskConfigs be de- and re-serializable, even when using the !function constructor?

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self):
        """dumps the current config as a dictionary object, as a printable format.
        null fields will not be printed.
        Used for dumping results alongside full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        # TODO: should any default value in the TaskConfig not be printed?
        """
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif isinstance(v, Callable):
                # TODO: this should handle Promptsource template objects as a separate case?
                cfg_dict[k] = str(v)
        return cfg_dict


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """

    VERSION = None

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    OUTPUT_TYPE: str = None

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        """
        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None
        self._instances = None

        self._config = TaskConfig({**config}) if config else TaskConfig()

        self._filters = [build_filter_ensemble("none", [["take_first", None]])]

    def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
        """Downloads and returns the task dataset.
        Override this method to download the dataset from a custom API.

        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        self.dataset_no_image = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        for doc_name in self.dataset_no_image:
            remove_cols = []
            features = self.dataset_no_image[doc_name].features
            # If it is an Image instance or a Sequence of Image instance. Remove it
            for feature in features:
                if isinstance(features[feature], Image):
                    remove_cols.append(feature)
                elif isinstance(features[feature], Sequence) and isinstance(features[feature].feature, Image):
                    remove_cols.append(feature)
            for remove_col in remove_cols:
                self.dataset_no_image[doc_name] = self.dataset_no_image[doc_name].remove_columns(remove_col)

    @property
    def config(self):
        """Returns the TaskConfig associated with this class."""
        return self._config

    @abc.abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass

    @abc.abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abc.abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    def training_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def fewshot_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        if self.has_training_docs():
            return self.training_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            if self.config.num_fewshot is not None:
                eval_logger.warning("has_training_docs and has_validation_docs are False" ", using test_docs as fewshot_docs but this is not recommended.")
            return self.test_docs()

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    @property
    def instances(self):
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc) -> None:
        print("Override doc_to_decontamination_query with document specific decontamination query.")
        assert False

    @abc.abstractmethod
    def doc_to_text(self, doc):
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc):
        pass

    # @profile
    def build_all_requests(self, limit=None, rank=None, world_size=None) -> None:
        """Build a set of Instances for a task, and store them in task.instances"""
        if self.has_test_docs():
            docs = self.test_docs()
            split = self.config.test_split
        elif self.has_validation_docs():
            docs = self.validation_docs()
            split = self.config.validation_split
        else:
            assert False, f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have valid or test docs!"

        eval_logger.info(f"Building contexts for task {self.CONFIG.task} on rank {rank}...")
        instances = []
        doc_id_iterator = utils.create_iterator([i for i in range(len(docs))], rank, world_size, limit)
        doc_id_iterator, doc_id_iterator_counting = itertools.tee(doc_id_iterator)
        total_docs = sum(1 for _ in doc_id_iterator_counting)
        pbar = tqdm(total=total_docs, desc=f"Building context", disable=(rank != 0))
        for doc_id in doc_id_iterator:
            # sample fewshot context #TODO: need to offset doc_id by rank now!
            fewshot_ctx = self.fewshot_context(doc_id, 0 if self.config.num_fewshot is None else self.config.num_fewshot, self.config.training_split if self.has_training_docs() else split)

            # TODO: we should override self.config.repeats if doing greedy gen so users don't waste time+compute
            inst = self.construct_requests(doc_id=doc_id, ctx=fewshot_ctx, metadata=(self.config["task"], doc_id, self.config.repeats), split=split)

            if not isinstance(inst, list):
                inst = [inst]

            instances.extend(inst)
            pbar.update(1)

        pbar.close()
        self._instances = instances
        assert len(self._instances) != 0, "task.build_requests() did not find any docs!"

    @abc.abstractmethod
    def construct_requests(self, doc_id, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LMM.

        :param doc_id: int
            The index of a document within `self.test_docs()` or `self.validation_docs()`,
            whichever is the main split used.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        :param repeats: int
        TODO: update this docstring
            The number of times each instance in a dataset is inferred on. Defaults to 1,
            can be increased for techniques like majority voting.
        """
        pass

    @abc.abstractmethod
    def process_results(self, doc, results):
        """Take a single document and the LMM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pass

    @abc.abstractmethod
    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        pass

    @abc.abstractmethod
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        pass

    @classmethod
    def count_bytes(cls, doc):
        """Used for byte-level perplexity metrics in rolling loglikelihood"""
        return len(doc.encode("utf-8"))

    @utils.positional_deprecated
    def fewshot_context(
        self,
        doc_id,
        num_fewshot,
        split,
        rnd=random.Random(1234),
        description=None,
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc_id: int
            The document id as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param split: str
            The split of the document to retrieve from the dataset
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"

        description = description if description else ""
        doc = self.dataset_no_image[split][doc_id]

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(self.validation_docs() if self.has_validation_docs() else self.test_docs())

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = "\n\n".join([self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshotex]) + "\n\n"

        example = self.doc_to_text(doc)
        return description + labeled_examples + example

    def apply_filters(self):
        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances, None)
        else:
            eval_logger.warning("No filter defined, passing through instances")
            return self._instances

    def dump_config(self) -> dict:
        """Returns a dictionary representing the task's config.

        :returns: str
            The fewshot context.
        """
        # TODO: this should only return the overrides applied to a non-YAML task's configuration.
        # (num_fewshot)
        return self.config.to_dict()


class ConfigurableTask(Task):
    VERSION = "Yaml"
    OUTPUT_TYPE = None
    CONFIG = None

    def __init__(self, model_name) -> None:  # TODO no super() call here
        # Get pre-configured attributes
        self._config = self.CONFIG
        # different model requires different prompt, we have to take those into account.

        self.model_name = model_name
        self._prepare_model_specific_config()

        assert self.config.output_type in ALL_OUTPUT_TYPES
        self.OUTPUT_TYPE = self.config.output_type

        self.DATASET_PATH = self.config.dataset_path

        if self.config.dataset_name is not None:
            self.DATASET_NAME = self.config.dataset_name

        self._prepare_metric_and_aggregation()

        self.download(self.config.dataset_kwargs)
        self._training_docs = None
        self._fewshot_docs = None

        if self.config.filter_list is not None:
            self._filters = []
            for filter_config in self.config.filter_list:
                for filter_pipeline in filter_config:
                    filter_name = filter_config["name"]
                    filter_functions = filter_config["filter"]
                    components = []
                    for function in filter_functions:
                        kwargs = {key: function[key] for key in function if key != "function"}
                        components.append([function["function"], kwargs])
                    filter_pipeline = build_filter_ensemble(filter_name, components)
                self._filters.append(filter_pipeline)
        else:
            self._filters = [build_filter_ensemble("none", [["take_first", None]])]
        if self.config.fewshot_config is not None:
            self.sampler = samplers.get_sampler(self.config.fewshot_config.get("sampler", "default") if self.config.fewshot_config else "default")(list(self.fewshot_docs()), self, rnd=random.Random(1234))

        if self.has_test_docs():
            self.task_docs = self.test_docs()
        elif self.has_validation_docs():
            self.task_docs = self.validation_docs()
        else:
            assert False, f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have valid or test docs!"

        # Test One Doc
        self.features = list(self.task_docs.features.keys())
        self.multiple_input = 0
        self.multiple_target = 0
        test_doc = self.task_docs[0]
        test_text = self.doc_to_text(test_doc)
        test_target = self.doc_to_target(test_doc)

        if self.config.doc_to_choice is not None:
            test_choice = self.doc_to_choice(test_doc)
            if type(test_choice) is not list:
                eval_logger.error("doc_to_choice must return list")
            else:
                num_choice = len(test_choice)

            if type(test_text) is int:
                self.multiple_input = num_choice
        else:
            test_choice = None

        if type(test_target) is list:
            self.multiple_target = len(test_target)
        else:
            if (type(test_target) is int) and (test_choice is not None):
                test_target = test_choice[test_target]
            else:
                test_target = str(test_target)

        if test_choice is not None:
            check_choices = test_choice
        else:
            check_choices = [test_target]
        if self.config.doc_to_choice is not None:
            for choice in check_choices:
                choice_has_whitespace = True if choice[0].isspace() else False
                delimiter_has_whitespace = True if self.config.target_delimiter.rstrip() != self.config.target_delimiter else False

                if delimiter_has_whitespace and choice_has_whitespace:
                    eval_logger.warning(f'Both target_delimiter and target choice: "{choice}" have whitespace')
                elif (not delimiter_has_whitespace) and (not choice_has_whitespace):
                    eval_logger.warning(f'Both target_delimiter "{self.config.target_delimiter}" and target choice: "{choice}" do not have whitespace, ignore if the language you are evaluating on does not require/use whitespace')

    def _prepare_model_specific_config(self):
        self.model_specific_prompt_kwargs = self.config.model_specific_prompt_kwargs
        if self.model_specific_prompt_kwargs is not None:
            if self.model_name in self.model_specific_prompt_kwargs:
                self.model_specific_prompt_kwargs = self.model_specific_prompt_kwargs[self.model_name]
            else:
                self.model_specific_prompt_kwargs = self.model_specific_prompt_kwargs.get("default", None)

        self.model_specific_target_kwargs = self.config.model_specific_target_kwargs
        if self.model_specific_target_kwargs is not None:
            if self.model_name in self.model_specific_target_kwargs:
                self.model_specific_target_kwargs = self.model_specific_target_kwargs[self.model_name]
            else:
                self.model_specific_target_kwargs = self.model_specific_target_kwargs.get("default", None)
        self.model_specific_generation_kwargs = self.config.model_specific_generation_kwargs
        if self.model_specific_generation_kwargs is not None:
            if self.model_name in self.model_specific_generation_kwargs:
                self.model_specific_generation_kwargs = self.model_specific_generation_kwargs[self.model_name]
            else:
                self.model_specific_generation_kwargs = self.model_specific_generation_kwargs.get("default", {})

            self.config.generation_kwargs.update(self.model_specific_generation_kwargs)

    def _prepare_metric_and_aggregation(self):
        self._metric_fn_list = {}
        self._metric_fn_kwargs = {}
        self._aggregation_list = {}
        self._higher_is_better = {}

        if self.config.metric_list is None:
            # TODO: handle this in TaskConfig.__post_init__ ?
            _metric_list = DEFAULT_METRIC_REGISTRY[self.config.output_type]

            for metric_name in _metric_list:
                self._metric_fn_list[metric_name] = METRIC_REGISTRY[metric_name]
                self._metric_fn_kwargs[metric_name] = {}
                self._aggregation_list[metric_name] = get_metric_aggregation(metric_name)
                self._higher_is_better[metric_name] = is_higher_better(metric_name)
        else:
            for metric_config in self.config.metric_list:
                assert "metric" in metric_config
                metric_name = metric_config["metric"]
                kwargs = {key: metric_config[key] for key in metric_config if key not in ["metric", "aggregation", "higher_is_better"]}

                if self.config.process_results is not None:
                    self._metric_fn_list[metric_name] = None
                    self._metric_fn_kwargs[metric_name] = {}
                elif callable(metric_name):
                    metric_fn = metric_name.__call__
                    metric_name = metric_name.__name__
                    self._metric_fn_list[metric_name] = metric_fn
                    self._metric_fn_kwargs[metric_name] = kwargs
                else:
                    self._metric_fn_list[metric_name] = METRIC_REGISTRY[metric_name]
                    self._metric_fn_kwargs[metric_name] = kwargs

                if "aggregation" in metric_config:
                    agg_name = metric_config["aggregation"]
                    if type(agg_name) == str:
                        self._aggregation_list[metric_name] = get_aggregation(agg_name)
                    elif callable(agg_name):
                        self._aggregation_list[metric_name] = metric_config["aggregation"]
                else:
                    INV_AGG_REGISTRY = {v: k for k, v in AGGREGATION_REGISTRY.items()}
                    metric_agg = get_metric_aggregation(metric_name)
                    eval_logger.warning(f"[Task: {self._config.task}] metric {metric_name} is defined, but aggregation is not. " f"using default " f"aggregation={INV_AGG_REGISTRY[metric_agg]}")
                    self._aggregation_list[metric_name] = metric_agg

                if "higher_is_better" in metric_config:
                    self._higher_is_better[metric_name] = metric_config["higher_is_better"]
                else:
                    eval_logger.warning(f"[Task: {self._config.task}] metric {metric_name} is defined, but higher_is_better is not. " f"using default " f"higher_is_better={is_higher_better(metric_name)}")
                    self._higher_is_better[metric_name] = is_higher_better(metric_name)

    @retry(stop=(stop_after_attempt(5) | stop_after_delay(60)), wait=wait_fixed(2))
    def download(self, dataset_kwargs=None) -> None:
        # If the dataset is a video dataset,
        # Recursively search whether their is a zip and unzip it to the huggingface home
        if dataset_kwargs is not None and "video" in dataset_kwargs and dataset_kwargs["video"]:
            hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
            cache_dir = dataset_kwargs["cache_dir"]
            cache_dir = os.path.join(hf_home, cache_dir)
            accelerator = Accelerator()
            if accelerator.is_main_process:
                cache_path = snapshot_download(repo_id=self.DATASET_PATH, repo_type="dataset")
                zip_files = glob(os.path.join(cache_path, "**/*.zip"), recursive=True)

                if not os.path.exists(cache_dir) and len(zip_files) > 0:
                    for zip_file in zip_files:
                        eval_logger.info(f"Unzipping {zip_file} to {cache_dir}")
                        shutil.unpack_archive(zip_file, cache_dir)

            accelerator.wait_for_everyone()

            if "builder_script" in dataset_kwargs:
                builder_script = dataset_kwargs["builder_script"]
                self.DATASET_PATH = os.path.join(cache_path, builder_script)
                dataset_kwargs.pop("builder_script")

            dataset_kwargs.pop("cache_dir")
            dataset_kwargs.pop("video")
        download_config = DownloadConfig()
        download_config.max_retries = dataset_kwargs.get("max_retries", 3) if dataset_kwargs is not None else 3
        download_config.num_proc = dataset_kwargs.get("num_proc", 8) if dataset_kwargs is not None else 8
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
            **dataset_kwargs if dataset_kwargs is not None else {},
        )
        self.dataset_no_image = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
            **dataset_kwargs if dataset_kwargs is not None else {},
        )
        for doc_name in self.dataset_no_image:
            remove_cols = []
            features = self.dataset_no_image[doc_name].features
            # If it is an Image instance or a Sequence of Image instance. Remove it
            for feature in features:
                if isinstance(features[feature], Image):
                    remove_cols.append(feature)
                elif isinstance(features[feature], Sequence) and isinstance(features[feature].feature, Image):
                    remove_cols.append(feature)
            for remove_col in remove_cols:
                self.dataset_no_image[doc_name] = self.dataset_no_image[doc_name].remove_columns(remove_col)

    def has_training_docs(self) -> bool:
        if self.config.training_split is not None:
            return True
        else:
            return False

    def has_validation_docs(self) -> bool:
        if self.config.validation_split is not None:
            return True
        else:
            return False

    def has_test_docs(self) -> bool:
        if self.config.test_split is not None:
            return True
        else:
            return False

    def training_docs(self) -> datasets.Dataset:
        if self.has_training_docs():
            if self.config.process_docs is not None:
                return self.config.process_docs(self.dataset[self.config.training_split])
            return self.dataset[self.config.training_split]

    def validation_docs(self) -> datasets.Dataset:
        if self.has_validation_docs():
            if self.config.process_docs is not None:
                return self.config.process_docs(self.dataset[self.config.validation_split])
            return self.dataset[self.config.validation_split]

    def test_docs(self) -> datasets.Dataset:
        if self.has_test_docs():
            if self.config.process_docs is not None:
                return self.config.process_docs(self.dataset[self.config.test_split])
            return self.dataset[self.config.test_split]

    def fewshot_docs(self):
        if self.config.fewshot_split is not None:
            return self.dataset[self.config.fewshot_split]
        else:
            if (self.config.num_fewshot is not None) and (self.config.num_fewshot > 0):
                eval_logger.warning(f"Task '{self.config.task}': " "num_fewshot > 0 but fewshot_split is None. " "using preconfigured rule.")
            return super().fewshot_docs()

    @utils.positional_deprecated
    def fewshot_context(self, doc_id, num_fewshot, split):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc_id: str
            The document id as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :returns: str
            The fewshot context.
        """
        doc = self.dataset_no_image[split][doc_id]
        if num_fewshot == 0:
            # always prepend the (possibly empty) task description
            labeled_examples = self.config.description
        else:
            labeled_examples = self.config.description + self.sampler.get_context(doc, num_fewshot)
        example = self.doc_to_text(doc)
        if type(example) == str:
            return labeled_examples + example
        elif type(example) == list:
            return [labeled_examples + ex for ex in example]
        elif type(example) == int:
            if self.config.doc_to_choice is not None:
                choices = self.doc_to_choice(doc)
                return labeled_examples + choices[example]
            else:
                return labeled_examples + str(example)

    def apply_filters(self):
        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances, self.task_docs)
        else:
            eval_logger.warning("No filter defined, passing through instances")
            return self._instances

    def should_decontaminate(self):
        return self.config.should_decontaminate

    def doc_to_decontamination_query(self, doc):
        if self.config.should_decontaminate:
            if self.config.doc_to_decontamination_query is None:
                return self.doc_to_text(doc)
            else:
                doc_to_decontamination_query = self.config.doc_to_decontamination_query
                if doc_to_decontamination_query in self.features:
                    return doc[doc_to_decontamination_query]
                elif callable(doc_to_decontamination_query):
                    return doc_to_decontamination_query(doc)
                else:
                    return ast.literal_eval(utils.apply_template(self.config.doc_to_decontamination_query, doc))

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    def doc_to_text(self, doc):
        doc_to_text = self.config.doc_to_text

        if type(doc_to_text) == int:
            return doc_to_text
        elif type(doc_to_text) == str:
            if doc_to_text in self.features:
                # if self.config.doc_to_choice is not None:
                #     return self.doc_to_choice(doc)[doc[doc_to_text]]
                # else:
                return doc[doc_to_text]
            else:
                text_string = utils.apply_template(doc_to_text, doc)
                if text_string.isdigit() and self._config.doc_to_choice is not None:
                    return ast.literal_eval(text_string)
                else:
                    return text_string
        elif callable(doc_to_text):
            return (
                doc_to_text(doc, self.model_specific_prompt_kwargs)
                if self.model_specific_prompt_kwargs is not None
                else doc_to_text(
                    doc,
                )
            )
        # Used when applying a Promptsource template
        elif hasattr(doc_to_text, "apply"):
            applied_prompt = doc_to_text.apply(doc)
            if len(applied_prompt) == 2:
                return applied_prompt[0]
            else:
                eval_logger.warning("Applied prompt returns empty string")
                return self.config.fewshot_delimiter
        else:
            print(type(doc_to_text))
            raise TypeError

    def doc_to_target(self, doc: dict) -> Union[int, str, list]:
        doc_to_target = self.config.doc_to_target

        if type(doc_to_target) == int:
            return doc_to_target
        elif type(doc_to_target) == str:
            if doc_to_target in self.features:
                # if self.config.doc_to_choice is not None:
                #     return self.doc_to_choice(doc)[doc[doc_to_target]]
                # else:
                return doc[doc_to_target]
            else:
                target_string = utils.apply_template(doc_to_target, doc)
                if target_string.isdigit() and self._config.doc_to_choice is not None:
                    return ast.literal_eval(target_string)
                elif len(target_string) >= 2 and (target_string[0] == "[") and (target_string[-1] == "]"):
                    try:
                        return ast.literal_eval(target_string)
                    except (SyntaxError, ValueError):
                        return target_string
                else:
                    return target_string
        elif type(doc_to_target) == list:
            return doc_to_target
        elif callable(doc_to_target):
            return doc_to_target(doc, self.model_specific_target_kwargs) if self.model_specific_target_kwargs is not None else doc_to_target(doc)
        # Used when applying a Promptsource template
        elif hasattr(doc_to_target, "apply"):
            applied_prompt = doc_to_target.apply(doc)
            if len(applied_prompt) == 2:
                return applied_prompt[1]
            else:
                eval_logger.warning("Applied prompt returns empty string")
                return self.config.fewshot_delimiter
        else:
            raise TypeError

    def doc_to_visual(self, doc: dict) -> Union[int, str, list]:
        self.config.doc_to_visual
        if type(self.config.doc_to_visual) == str:
            assert self.config.doc_to_visual in self.features
            # Single image. Still return a list for consistency.
            return [doc[self.config.doc_to_visual]]
        else:
            assert callable(self.config.doc_to_visual)
            return self.config.doc_to_visual(doc)

    def doc_to_choice(self, doc: Any) -> List[str]:
        if self.config.doc_to_choice is None:
            eval_logger.error("doc_to_choice was called but not set in config")
        else:
            doc_to_choice = self.config.doc_to_choice

        if type(doc_to_choice) == str:
            if doc_to_choice in self.features:
                return doc[doc_to_choice]
            else:
                return ast.literal_eval(utils.apply_template(doc_to_choice, doc))
        elif type(doc_to_choice) == list:
            return doc_to_choice
        elif type(doc_to_choice) == dict:
            return list(doc_to_choice.values())
        elif callable(doc_to_choice):
            return doc_to_choice(doc)
        elif hasattr(doc_to_choice, "get_answer_choices_list"):
            return doc_to_choice.get_answer_choices_list(doc)
        else:
            raise TypeError

    def construct_requests(self, doc_id: int, ctx: str, **kwargs) -> Union[List[Instance], Instance]:
        split = kwargs.get("split")
        kwargs.pop("split")
        if self.OUTPUT_TYPE == "loglikelihood":
            arguments = (ctx, self.doc_to_target, self.doc_to_visual, doc_id, self.config.task, split)
        elif self.OUTPUT_TYPE == "multiple_choice":
            doc = self.dataset[split][doc_id]
            choices = self.doc_to_choice(doc)
            target_delimiter = self.config.target_delimiter
            if self.multiple_input:
                # If there are multiple inputs, choices are placed in the ctx
                cont = self.doc_to_target(doc)
                arguments = [(ctx, f"{target_delimiter}{cont}", self.doc_to_visual, doc_id, self.config.task, split) for ctx in choices]
            else:
                # Otherwise they are placed in the continuation
                arguments = [(ctx, f"{target_delimiter}{cont}", self.doc_to_visual, doc_id, self.config.task, split) for cont in choices]
            request_list = [
                Instance(
                    request_type="loglikelihood",
                    # doc=doc,
                    arguments=arg,
                    idx=i,
                    **kwargs,
                )
                for i, arg in enumerate(arguments)
            ]
            # TODO: we should raise a warning telling users this will at most ~2x runtime.
            if "acc_mutual_info" in self._metric_fn_list.keys():
                # if we are calculating multiple choice accuracy
                # using mutual information instead of raw loglikelihood as metric, need unconditional lls.

                # here mutual info refers to calculating
                # log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
                # in other words normalizing by subtracting the unconditional logprob of each choice.
                request_list.extend(
                    [
                        Instance(
                            request_type="loglikelihood",
                            # doc=doc,
                            arguments=("", "{}".format(choice)),
                            idx=i,
                            **kwargs,
                        )
                        for i, choice in enumerate(choices)
                    ]
                )
            return request_list

        elif self.OUTPUT_TYPE == "generate_until":
            arguments = (ctx, self.config.generation_kwargs, self.doc_to_visual, doc_id, self.config.task, split)
        return Instance(request_type=self.OUTPUT_TYPE, arguments=arguments, idx=0, **kwargs)

    # TODO: we add a full_docs interface here for some evaluations that needs to access the full datasets during process_results function. we may have better ways to handle this.
    @retry(stop=(stop_after_attempt(5) | stop_after_delay(1200)), wait=wait_fixed(2))
    def process_results(self, doc, results, full_docs=None):
        if self.OUTPUT_TYPE == "generate_until":
            results[0] = results[0].strip()

        kwargs = {}
        if full_docs is not None:
            kwargs["full_docs"] = full_docs
        if callable(self.config.process_results):
            return self.config.process_results(doc, results, **kwargs)

        result_dict = {}
        use_metric = list(self._metric_fn_list.keys())
        if self.OUTPUT_TYPE == "loglikelihood":
            results = results[0]
            ll, is_greedy = results
            return {
                **({"perplexity": ll} if "perplexity" in use_metric else {}),
                **({"acc": int(is_greedy)} if "acc" in use_metric else {}),
            }
        elif self.OUTPUT_TYPE == "multiple_choice":
            lls, is_greedy = zip(*results)

            # retrieve choices in List[str] form, to compute choice lengths, etc.
            choices = self.doc_to_choice(doc)
            completion_len = np.array([float(len(i)) for i in choices])

            if 2 * len(choices) == len(lls) and "acc_mutual_info" in self._metric_fn_list.keys():
                # then we are doing mutual info.
                # this stores the "dryrun" / unconditional answer loglikelihoods
                lls_unconditional = lls[1::2]
                assert len(lls_unconditional) == len(choices)
                # and this stores our "regular" conditional loglikelihoods
                lls = lls[::2]

            pred = np.argmax(lls)
            pred_norm = np.argmax(lls / completion_len)

            if self.multiple_input:
                gold = self.doc_to_text(doc)
            else:
                gold = self.doc_to_target(doc)

            gold_index_error = False
            if type(gold) is list:
                gold = [i if i < len(choices) else -100 for i in gold]
                if -100 in gold:
                    gold_index_error = True
            else:
                if type(gold) is int:
                    gold = gold if gold < len(choices) else -100
                elif type(gold) is str:
                    gold = choices.index(gold) if gold in choices else -100

                if gold == -100:
                    gold_index_error = True

            if gold_index_error:
                eval_logger.warning(f"Label index was not in within range of available choices," f"Sample:\n\n{doc}\n\n")

            if self.multiple_target:
                acc = 1.0 if pred in gold else 0.0
                acc_norm = 1.0 if pred_norm in gold else 0.0
                exact_match = int(any([is_greedy[i] if i != -100 else 0 for i in gold]))
            else:
                acc = 1.0 if pred == gold else 0.0
                acc_norm = 1.0 if pred_norm == gold else 0.0
                # TODO: this gets score of 0 on arc_challenge for pythia-70m. need to test that this works properly
                exact_match = int(is_greedy[gold]) if gold != -100 else 0

            result_dict = {
                **({"acc": acc} if "acc" in use_metric else {}),
                **({"f1": (gold, pred)} if "f1" in use_metric else {}),
                **({"mcc": (gold, pred)} if "mcc" in use_metric else {}),
                **({"acc_norm": acc_norm} if "acc_norm" in use_metric else {}),
                **({"exact_match": exact_match} if "exact_match" in use_metric else {}),
            }

            if "acc_mutual_info" in use_metric:
                lls_mutual_info = [ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional)]
                acc_mutual_info = 1.0 if np.argmax(lls_mutual_info) == gold else 0.0
                result_dict["acc_mutual_info"] = acc_mutual_info

        elif self.OUTPUT_TYPE == "generate_until":
            gold = self.doc_to_target(doc)
            result = results[0]
            if self.config.doc_to_choice is not None:
                # If you set doc_to_choice,
                # it assumes that doc_to_target returns a number.
                choices = self.doc_to_choice(doc)
                gold = choices[gold]
            # we expect multiple_targets to be a list.
            elif self.multiple_target:
                gold = list(gold)
            elif type(gold) != type(result):
                # cast gold to the same type as result
                gold = type(result)(gold)

            for metric in self._metric_fn_list.keys():
                if self.multiple_target:
                    # in the case where we have multiple targets,
                    # return true if any are true
                    # TODO: this may break for multipLe_target, non zero-or-1 metrics
                    scores = []
                    if not isinstance(gold, list):
                        # sometimes, a multiple_target dataset has exceptions where one doc has only one string answer
                        # print(gold)
                        gold = [gold]
                    for gold_option in gold:
                        try:
                            result_score = self._metric_fn_list[metric](
                                references=[gold_option],
                                predictions=[result],
                                **self._metric_fn_kwargs[metric],
                            )
                        except TypeError:  # TODO: this is hacky and I don't want to do it
                            result_score = self._metric_fn_list[metric]([gold_option, result])
                        if isinstance(result_score, dict):
                            # TODO: this handles the case where HF evaluate returns a dict.
                            result_score = result_score[metric]
                        scores.append(result_score)
                    if any(scores):
                        result_score = 1.0
                    else:
                        result_score = 0.0
                else:
                    try:
                        result_score = self._metric_fn_list[metric](
                            references=[gold],
                            predictions=[result],
                            **self._metric_fn_kwargs[metric],
                        )
                    except TypeError:  # needed for now in order to use a different interface between our own metrics and HF Evaluate metrics
                        result_score = self._metric_fn_list[metric]([gold, result])
                    if isinstance(result_score, dict):
                        # TODO: this handles the case where HF evaluate returns a dict.
                        result_score = result_score[metric]
                result_dict[metric] = result_score
        else:
            raise ValueError(
                f"Passed invalid output_type '{self.OUTPUT_TYPE}' ! Please use one of ",
                "'loglikelihood','generate_until' or 'multiple_choice'",
            )

        return result_dict

    def aggregation(self):
        return self._aggregation_list

    def higher_is_better(self):
        return self._higher_is_better

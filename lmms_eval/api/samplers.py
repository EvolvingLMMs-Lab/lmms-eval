import datasets
from typing import Callable


class LazyLoadedImages(object):
    def __init__(self, data_frame, index):
        self.data_frame: datasets.Dataset = data_frame
        self.index = index

    def get_images(self, doc_to_visual):
        return doc_to_visual(self.data_frame[self.index])


class Context(object):
    def __init__(self, task, few_shot_delimiter: str = "\n\n", target_delimiter: str = "\n", description = None):
        self.task = task
        self.config = task._config

        self.doc_to_visual = self.task.doc_to_visual
        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.target_delimiter = target_delimiter
        self.few_shot_delimiter = few_shot_delimiter

        self.contexts = []
        
        if description:
            self.add_description(description)
    
    def add_description(self, description):
        self.contexts = [description] + self.contexts

    def get_question(self, doc):
        text = self.doc_to_text(doc)
        return text if (self.doc_to_choice is None or isinstance(text, str)) else self.doc_to_choice(doc)[text]

    def get_target(self, doc):
        return (
            str(self.doc_to_target(doc)[0])
            if type(self.doc_to_target(doc)) is list
            else self.doc_to_target(doc) if (self.config.doc_to_choice is None or type(self.doc_to_target(doc)) is str) else str(self.doc_to_choice(doc)[self.doc_to_target(doc)])
        )

    def add_in_context_example(self, doc, data_frame=None, index=None):
        question = self.get_question(doc)
        if data_frame and index:
            visual = LazyLoadedImages(data_frame, index)
        else:
            visual = None
        target = self.doc_to_target(doc)
        if visual:
            self.contexts.append(visual)
        self.contexts.append(question)
        self.contexts.append(self.target_delimiter)
        self.contexts.append(target)
        self.contexts.append(self.few_shot_delimiter)

    def add_question(self, doc, data_frame=None, index=None):
        question = self.get_question(doc)
        if data_frame and index:
            visual = LazyLoadedImages(data_frame, index)
        else:
            visual = None
        if visual:
            self.contexts.append(visual)
        self.contexts.append(question)
        self.contexts.append(self.target_delimiter)


class FewShotDataset(object):
    def __init__(self, dataset=None, *, dataset_path: str = None, dataset_name: str = None, split: str = None, dataset_kwargs: dict = None, same_as_eval: bool = False):
        if dataset is not None and (dataset_path is not None or dataset_name is not None or split is not None or dataset_kwargs is not None):
            raise ValueError("Cannot provide both `dataset` and other dataset arguments!")
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = dataset
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}
        self.same_as_eval = same_as_eval
        self.fewshot_indices = None

    def get_dataset(self) -> datasets.Dataset:
        if self.dataset is None:
            self.dataset = datasets.load_dataset(path=self.dataset_path, name=self.dataset_name, split=self.split, download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, **self.dataset_kwargs)
            if self.fewshot_indices:
                self.dataset = self.dataset.select(self.fewshot_indices)
        return self.dataset

    def sample(self, n, rnd):
        indices = rnd.sample(range(len(self.get_dataset())), n)
        return indices, self.get_dataset().select(indices)

    def __getitem__(self, item):
        return self.get_dataset()[item]


class ContextSampler:
    def __init__(self, docs: FewShotDataset, task, fewshot_indices=None, rnd=None) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs: FewShotDataset = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs.fewshot_indices = fewshot_indices

    def get_context(self, doc, num_fewshot) -> Context:
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = num_fewshot + 1 if self.docs.same_as_eval else num_fewshot

        # draw `n_samples` docs from fewshot_docs
        indices, fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [(idx, x) for idx, x in zip(indices, fewshotex) if x != doc][:num_fewshot]

        labeled_examples = Context(self.task, self.fewshot_delimiter, self.target_delimiter)

        for idx, doc in selected_docs:
            labeled_examples.add_in_context_example(doc, self.docs, idx)

        return labeled_examples

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.docs.sample(n, self.rnd)


class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.docs), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, n) -> None:
        """ """
        pass


SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}")

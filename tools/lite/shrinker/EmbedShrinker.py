from .BaseShrinker import BaseShrinker
import sys
from .sampling_methods import AVAILABEL_METHODS
from lmms_eval.tasks import initialize_tasks

import torch
from typing import List, Dict, Union
import numpy as np
import os

sys.path.append("..")
from embedder import BaseEmbedder
from shrinker import sampling_methods as sampling_methods_module


class Embed_Shrinker(BaseShrinker):
    def __init__(
        self,
        task: str,
        num_items: Union[int, float],
        name: str,
        embed_cache_path: str,
        sampling_methods: str,
        push_to_hub: bool,
    ) -> None:
        super().__init__(task, num_items, name, push_to_hub)
        self.embed_cache_path = embed_cache_path
        initialize_tasks()
        self.DATASET_PATH, self.DATASET_NAME, self.split, _, self.task_obj, docs = BaseEmbedder.init_task(task)
        assert sampling_methods in AVAILABEL_METHODS, f"Not available sampling methods, Choose from {AVAILABEL_METHODS.keys()}"
        self.sampling_methods = getattr(sampling_methods_module, AVAILABEL_METHODS[sampling_methods])

    def shrink(self):
        task_embedding = np.load(open(os.path.join(self.embed_cache_path, f"{self.task}_embed.npy"), "rb"))
        task_embedding = torch.from_numpy(task_embedding)
        # I know torch.squeeze is safe but numpy reshape sometimes may not
        # so I just do it here by converting to torch
        if len(task_embedding.shape) == 3:
            task_embedding = task_embedding.squeeze(1)
        task_embedding = task_embedding.numpy()
        self.sampling_methods = self.sampling_methods(X=task_embedding)
        # centroids = self.cluster(task_embedding)
        if self.num_items < 1.0:
            self.num_items = int(task_embedding.shape[0] * self.num_items)
        else:
            self.num_items = int(self.num_items)
        anchor_points = self.sampling_methods.select_batch(N=self.num_items)
        dataset = self.task_obj.dataset[self.split]
        tiny_dataset = dataset.select(anchor_points)

        if self.push_to_hub:
            tiny_dataset.push_to_hub(repo_id=f"lmms-lab/LMMs-Eval-Lite", config_name=self.task, split="lite")
        return

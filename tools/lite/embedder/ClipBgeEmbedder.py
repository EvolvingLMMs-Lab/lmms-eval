import json
import os
from copy import deepcopy
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .BaseEmbedder import BaseEmbedder


class ClipBgeEmbedder(BaseEmbedder):
    def __init__(
        self,
        name: str,
        output_path: str,
        mm_pretrained: str = "openai/clip-vit-large-patch14",
        txt_pretrained: str = "BAAI/bge-m3",
        device: str = "cuda",
        device_map: str = "",
    ) -> None:
        super().__init__(name, output_path)
        self.model = CLIPModel.from_pretrained(mm_pretrained)
        self.processor = CLIPProcessor.from_pretrained(mm_pretrained)
        self.text_model = SentenceTransformer(txt_pretrained)
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if self.accelerator.num_processes > 1 and device_map == "":
            self.device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            self.device_map = f"cuda:{self.accelerator.local_process_index}"
        else:
            self.device = torch.device(device)
            self.device_map = device_map

        self.model.to(self.device)
        self.text_model.to(self.device)

    def embed_task(self, task: str, ignored_ids: Union[set, List] = None):
        DATASET_PATH, DATASET_NAME, split, requests, task_obj, self.docs = BaseEmbedder.init_task(task, ignored_ids)
        self.accelerator.wait_for_everyone()
        with self.accelerator.split_between_processes(requests, apply_padding=False) as requests_split:
            results = {"outputs": []}
            for req in tqdm(requests_split, disable=not self.accelerator.is_main_process):
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = req.args
                visuals = [doc_to_visual(self.docs[doc_id])]
                visuals = self.flatten(visuals)

                text_embedding = self.text_model.encode([contexts])
                text_embedding = torch.from_numpy(text_embedding).flatten()

                if len(visuals) > 0:
                    img_inputs = self.processor(images=visuals, return_tensors="pt")
                    img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}

                    # For multiple images, we take the mean of it
                    image_embedding = self.model.get_image_features(**img_inputs).mean(dim=0).detach().cpu()
                else:
                    image_embedding = torch.zeros(self.model.config.projection_dim)

                embedding = torch.concat([image_embedding, text_embedding])

                results["outputs"].append(embedding)
            results = [results]

        self.accelerator.wait_for_everyone()
        results_gathered = gather_object(results)
        if self.accelerator.is_main_process:
            outputs = []
            for r in results_gathered:
                outputs += r["outputs"]
            results_gathered = torch.stack(outputs)
            np.save(open(os.path.join(self.output_path, f"{task}_embed.npy"), "wb"), results_gathered)
            return results_gathered


if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    text = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.get_text_features(**inputs)
    print(outputs.mean(dim=0).shape)

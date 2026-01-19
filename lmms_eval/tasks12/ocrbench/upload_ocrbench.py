# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import io
import json
from uuid import uuid4

import datasets
import pandas as pd
from datasets import Dataset, Features
from PIL import Image as PIL_Image
from tqdm import tqdm

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """https://arxiv.org/abs/2305.07895"""
_DESCRIPTION = "OCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models."


def image2byte(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    image_bytes = img_bytes.getvalue()
    return image_bytes


def get_builder_config(VERSION):
    builder_config = [
        datasets.BuilderConfig(
            name=f"ocrbench",
            version=VERSION,
            description=f"ocrbench",
        )
    ]
    return builder_config


ocrbench_json = "pathto/OCRBench/OCRBench.json"
img_dir = "pathto/OCRBench_Images/"

dataset_features = Features(
    {
        "dataset": datasets.Value("string"),
        "question": datasets.Value("string"),
        "question_type": datasets.Value("string"),
        "answer": datasets.features.Sequence(datasets.Value("string")),
        "image": datasets.Image(),
    }
)

df_items = {
    "dataset": [],
    "question": [],
    "question_type": [],
    "answer": [],
    "image": [],
}
# img_feature = datasets.Image(decode=False)
with open(ocrbench_json, "r") as f:
    data = json.load(f)
for i in tqdm(range(len(data))):
    dataset_name = data[i]["dataset_name"]
    image_path = img_dir + data[i]["image_path"]
    question = data[i]["question"]
    answers = data[i]["answers"]
    question_type = data[i]["type"]
    if type(answers) == str:
        answers = [answers]
    img = PIL_Image.open(image_path).convert("RGB")
    byte_data = image2byte(img)
    image = {"bytes": byte_data, "path": ""}
    df_items["image"].append(image)
    df_items["question"].append(str(question))
    df_items["answer"].append(answers)
    df_items["question_type"].append(str(question_type))
    df_items["dataset"].append(str(dataset_name))

df_items = pd.DataFrame(df_items)
df_items.head()
dataset = Dataset.from_pandas(df_items, features=dataset_features)
hub_dataset_path = "echo840/OCRBench"
dataset.push_to_hub(repo_id=hub_dataset_path, split="test")

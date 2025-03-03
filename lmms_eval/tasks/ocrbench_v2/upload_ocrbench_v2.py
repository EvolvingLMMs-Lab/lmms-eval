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
import ipdb
from uuid import uuid4

import datasets
import pandas as pd
from datasets import Dataset, Features
from PIL import Image as PIL_Image
from tqdm import tqdm

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """https://www.arxiv.org/abs/2501.00321"""
_DESCRIPTION = "OCRBench v2 is a comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models."


def image2byte(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    image_bytes = img_bytes.getvalue()
    return image_bytes


def get_builder_config(VERSION):
    builder_config = [
        datasets.BuilderConfig(
            name=f"OCRBench v2",
            version=VERSION,
            description=f"OCRBench v2",
        )
    ]
    return builder_config


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, int):
        return str(obj)
    return obj


ocrbench_json = "DataPath/OCRBench_v2/OCRBench_v2.json"
img_dir = "DataPath/OCRBench_v2/"

dataset_features = Features(
    {
        "id": datasets.Value("int32"),
        "dataset_name": datasets.Value("string"),
        "question": datasets.Value("string"),
        "type": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "image": datasets.Image(),
        "eval": datasets.Value("string"),
        "bbox": datasets.features.Sequence(datasets.Value("int32")),
        "bbox_list": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32"))),
        "content": datasets.features.Sequence(datasets.Value("string")),
    }
)

df_items = {
    "id": [],
    "dataset_name": [],
    "question": [],
    "type": [],
    "answers": [],
    "image": [],
    "eval": [],
    "bbox": [],
    "bbox_list": [],
    "content": [],
}
# img_feature = datasets.Image(decode=False)
with open(ocrbench_json, "r") as f:
    data = json.load(f)

for i in tqdm(range(len(data))):
# for i in tqdm(range(0, len(data), 100)):
    id = data[i]["id"]
    dataset_name = data[i]["dataset_name"]
    image_path = img_dir + data[i]["image_path"]
    question = data[i]["question"]
    answers = data[i].get("answers", [])
    task_type = data[i].get("type", None)
    
    answers = [convert_to_serializable(a) for a in answers]

    eval_type = data[i].get("eval", None)
    
    bbox = data[i].get("bbox", None)
    if task_type == "text spotting en":
        bbox_list = bbox
        bbox = None
    else:
        bbox_list = None

    content = data[i].get("content", None)

    img = PIL_Image.open(image_path).convert("RGB")
    byte_data = image2byte(img)
    image = {"bytes": byte_data, "path": ""}
    df_items["id"].append(int(id))
    df_items["image"].append(image)
    df_items["question"].append(str(question))
    df_items["answers"].append(answers)
    df_items["type"].append(str(task_type))
    df_items["dataset_name"].append(str(dataset_name))
    df_items["eval"].append(str(eval_type))
    df_items["bbox"].append(bbox)
    df_items["bbox_list"].append(bbox_list)
    df_items["content"].append(content)

df_items = pd.DataFrame(df_items)
df_items.head()
dataset = Dataset.from_pandas(df_items, features=dataset_features)
hub_dataset_path = "xx"

dataset.push_to_hub(repo_id=hub_dataset_path, split="test")

ipdb.set_trace()

print("End of Code!")

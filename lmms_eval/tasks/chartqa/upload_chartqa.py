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


import json
import os
from PIL import Image
import datasets


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{masry-etal-2022-chartqa,
    title = "{C}hart{QA}: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning",
    author = "Masry, Ahmed  and
      Long, Do  and
      Tan, Jia Qing  and
      Joty, Shafiq  and
      Hoque, Enamul",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.177",
    doi = "10.18653/v1/2022.findings-acl.177",
    pages = "2263--2279",
}
"""
_DESCRIPTION = "A largescale benchmark covering 9.6K human-written questions as well as 23.1K questions generated from human-written chart summaries."


def get_builder_config(VERSION):
    builder_config = [
        datasets.BuilderConfig(
            name=f"ChartQA",
            version=VERSION,
            description=f"ChartQA",
        )
    ]
    return builder_config


dataset_features = {
    "type": datasets.Value("string"),
    "question": datasets.Value("string"),
    "answer": datasets.Value("string"),
    "image": datasets.Image(),
}


class ChartQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = get_builder_config(VERSION)

    def _info(self):
        features = datasets.Features(dataset_features)
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        image_path = "/home/yhzhang/ChartQA/ChartQA Dataset/test/png/"
        human_annotation_path = "/home/yhzhang/ChartQA/ChartQA Dataset/test/test_human.json"
        augmented_annotation_path = "/home/yhzhang/ChartQA/ChartQA Dataset/test/test_augmented.json"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "human_annotation": human_annotation_path,
                    "augmented_annotation": augmented_annotation_path,
                    "images": image_path,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, human_annotation, augmented_annotation, images):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(human_annotation, encoding="utf-8") as f:
            human_data = json.load(f)
        with open(augmented_annotation, encoding="utf-8") as f:
            augmented_data = json.load(f)
        index = -1
        for data in [human_data, augmented_data]:
            for row in data["data"]:
                index += 1
                image_path = os.path.join(images, row["imgname"])
                now_data = {}
                now_data["type"] = "human_test" if data == human_data else "augmented_test"
                now_data["image"] = Image.open(image_path)
                now_data["question"] = row["query"]
                now_data["answer"] = row["label"]
                yield index, now_data


if __name__ == "__main__":
    from datasets import load_dataset

    data = load_dataset(
        "/home/yhzhang/lmms-eval/lmms_eval/tasks/chartqa/upload_chartqa.py",
    )
    data.push_to_hub("lmms-lab/chartqa", private=True)

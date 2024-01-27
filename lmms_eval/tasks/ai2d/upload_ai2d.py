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

import datasets
from PIL import Image

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """https://allenai.org/data/diagrams"""
_DESCRIPTION = "AI2D is a dataset of illustrative diagrams for research on diagram understanding and associated question answering."


def get_builder_config(VERSION):
    builder_config = [
        datasets.BuilderConfig(
            name=f"ai2d",
            version=VERSION,
            description=f"ai2d",
        )
    ]
    return builder_config


dataset_features = {
    "question": datasets.Value("string"),
    "options": datasets.features.Sequence(datasets.Value("string")),
    "answer": datasets.Value("string"),
    "image": datasets.Image(),
}


class AI2D(datasets.GeneratorBasedBuilder):
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
        image_path = "/home/yhzhang/ai2d/images"
        annotation_path = "/home/yhzhang/ai2d/questions"
        # wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ai2diagram/test.jsonl
        test_annotation_path = "/home/yhzhang/Qwen-VL/data/ai2diagram/test.jsonl"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"annotation": annotation_path, "images": image_path, "test_annotation": test_annotation_path},
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, annotation, images, test_annotation):
        # open test_annotation jsonl (note; not a json)
        with open(test_annotation, encoding="utf-8") as f:
            test_annotation = [json.loads(line) for line in f]
        test_qn_ids = {x["question_id"] for x in test_annotation}
        index = -1
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        for sub_annotation in os.listdir(annotation):
            sub_annotation = os.path.join(annotation, sub_annotation)
            with open(sub_annotation, encoding="utf-8") as f:
                data = json.load(f)
            image = data["imageName"]
            image_path = os.path.join(images, image)
            for question in data["questions"]:
                if data["questions"]["questionId"] in test_qn_ids:
                    index += 1
                    options = data["questions"][question]["answerTexts"]
                    answer = data["questions"][question]["correctAnswer"]

                    now_data = {}
                    now_data["image"] = Image.open(image_path)
                    now_data["question"] = question
                    now_data["answer"] = answer
                    now_data["options"] = options
                    yield index, now_data


if __name__ == "__main__":
    from datasets import load_dataset

    data = load_dataset(
        "/home/yhzhang/lmms-eval/lmms_eval/tasks/ai2d/upload_ai2d.py",
    )
    data.push_to_hub("lmms-lab/ai2d", private=True)

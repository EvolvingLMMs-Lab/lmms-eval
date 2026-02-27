import datasets
from datasets import load_dataset
from tqdm import tqdm


def gqa_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    gqa_raw_image_dataset = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
    gqa_id2image = {}
    for row in tqdm(gqa_raw_image_dataset, desc="Loading GQA images"):
        gqa_id2image[row["id"]] = row["image"].convert("RGB")

    def _process_doc(doc):
        image = gqa_id2image[doc["imageId"]]
        return {
            "image": image,
        }

    return dataset.map(_process_doc, num_proc=8)


def gqa_doc_to_visual(doc):
    image = [doc["image"].convert("RGB")]
    return image


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

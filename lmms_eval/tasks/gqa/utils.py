from datasets import load_dataset
import os

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None


def gqa_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    
    if GQA_RAW_IMAGE_DATASET is None:
        local_infer_dataset_path = (
            lmms_eval_specific_kwargs.get("local_infer_dataset_path") 
            if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("local_infer_dataset_path") is not None 
            else None
        )
        dataset_path = os.path.join(local_infer_dataset_path, "lmms-lab/GQA") if local_infer_dataset_path else "lmms-lab/GQA"
        
        GQA_RAW_IMAGE_DATASET = load_dataset(
            path=dataset_path,
            name="testdev_balanced_images", 
            split="testdev", 
            token=True
        )
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

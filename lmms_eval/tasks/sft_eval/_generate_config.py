import os
import yaml

splits = [
    "sft-activity",
    "sft-arts",
    "sft-body",
    "sft-car",
    "sft-color",
    "sft-commodity",
    "sft-count",
    "sft-daily",
    "sft-engineer",
    "sft-entertainment",
    "sft-exist",
    "sft-face",
    "sft-food",
    "sft-healthcare",
    "sft-landmark",
    "sft-logo",
    "sft-natural",
    "sft-ocr_qa_adv",
    "sft-ocr_qa_chart",
    "sft-ocr_qa_form",
    "sft-ocr_qa_scene",
    "sft-ocr_qa_screen",
    "sft-ocr_rec_adv",
    "sft-ocr_rec_doc",
    "sft-ocr_rec_handwrite",
    "sft-ocr_rec_markdown",
    "sft-ocr_rec_scene",
    "sft-ocr_rec_screen",
    "sft-place",
    "sft-position",
    "sft-sport",
    "sft-status",
]
dir_path = os.path.dirname(os.path.realpath(__file__))

local_name2official_name = {
    "sft-ocr_rec_scene": "sft_ocr_scene_cn_eval",
    "sft-ocr_rec_screen": "sft_ocr_screen_cn_eval",
    "sft-ocr_rec_handwrite": "sft_ocr_handwrite_cn_eval",
    "sft-ocr_rec_adv": "sft_ocr_adv_cn_eval",
    "sft-ocr_rec_doc": "sft_ocr_doc_cn_eval",
    "sft-ocr_qa_scene": "sft_ocr_sceneQA_cn_eval",
    "sft-ocr_qa_screen": "sft_ocr_screenQA_cn_eval",
    "sft-ocr_qa_adv": "sft_ocr_ecommerceQA_cn_eval",
    "sft-ocr_rec_markdown": "sft_ocr_markdown_cn_eval",
    "sft-ocr_qa_form": "sft_ocr_formQA_cn_eval",
    "sft-ocr_qa_chart": "sft_ocr_chartQA_cn_eval",
    "sft-face": "sft_celeb_cn_eval",
    "sft-body": "sft_body_cn_eval",
    "sft-count": "sft_count_cn_eval",
    "sft-position": "sft_position_cn_eval",
    "#": "sft_visualprompt_cn_eval",
    "!": "sft_grounding_cn_eval",
    "sft-exist": "sft_exist_cn_eval",
    "sft-color": "sft_color_cn_eval",
    "sft-status": "sft_status_cn_eval",
    "sft-activity": "sft_activity_cn_eval",
    "sft-place": "sft_place_cn_eval",
    "sft-daily": "sft_daily_cn_eval",
    "sft-arts": "sft_arts_cn_eval",
    "sft-natural": "sft_natural_cn_eval",
    "sft-engineer": "sft_engineer_cn_eval",
    "sft-healthcare": "sft_healthcare_cn_eval",
    "sft-entertainment": "sft_entertainment_cn_eval",
    "sft-sport": "sft_sport_cn_eval",
    "sft-commodity": "sft_commodity_cn_eval",
    "sft-food": "sft_food_cn_eval",
    "sft-car": "sft_car_cn_eval",
    "sft-landmark": "sft_landmark_cn_eval",
}

task_name = [local_name2official_name[split] if split in local_name2official_name else split for split in splits]
splits = [split.replace("-", "_") for split in splits]

if __name__ == "__main__":
    for split, task in zip(splits, task_name):
        yaml_dict = {"group": f"sft_eval", "task": task, "test_split": split, "dataset_name": split}
        save_path = os.path.join(dir_path, f"{split}.yaml")
        if "ocr" in split and "rec" in split:
            yaml_dict["include"] = "_default_sft_eval_ocr_rec_template_yaml"
        else:
            yaml_dict["include"] = "_default_sft_eval_rest_template_yaml"
        print(f"Saving to {save_path}")
        with open(save_path, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    group_dict = {"group": "sft_eval", "task": splits}

    with open(os.path.join(dir_path, "_sft_eval.yaml"), "w") as f:
        yaml.dump(group_dict, f, default_flow_style=False, indent=4)

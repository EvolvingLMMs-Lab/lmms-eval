import os
import yaml

splits = ["train", "test", "val"]
tasks = ["seg", "bbox"]

if __name__ == "__main__":
    for task in tasks:
        for split in splits:
            yaml_dict = {"group": f"refcocog_{task}", "task": f"refcocog_{task}_{split}", "include": f"_default_template_{task}_yaml", "test_split": split}

            save_path = f"./refcocog_{task}_{split}.yaml"
            print(f"Saving to {save_path}")
            with open(save_path, "w") as f:
                yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    group_dict = {"group": "refcocog", "task": ["refcocog_bbox", "refcocog_seg"]}

    with open("./_refcoco.yaml", "w") as f:
        yaml.dump(group_dict, f, default_flow_style=False, indent=4)

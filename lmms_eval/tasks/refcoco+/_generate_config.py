import os
import yaml

splits = ["train", "val", "testA", "testB"]
tasks = ["seg", "bbox"]

if __name__ == "__main__":
    for task in tasks:
        for split in splits:
            yaml_dict = {"group": f"refcoco+_{task}", "task": f"refcoco+_{task}_{split}", "include": f"_default_template_{task}_yaml", "test_split": split}

            save_path = f"./refcoco+_{task}_{split}.yaml"
            print(f"Saving to {save_path}")
            with open(save_path, "w") as f:
                yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    group_dict = {"group": "refcoco+", "task": ["refcoco+_bbox", "refcoco+_seg"]}

    with open("./_refcoco.yaml", "w") as f:
        yaml.dump(group_dict, f, default_flow_style=False, indent=4)

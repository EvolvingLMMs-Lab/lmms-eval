import os

import yaml

splits = ["val", "test"]
tasks = ["vqa"]

if __name__ == "__main__":
    dump_tasks = []
    for task in tasks:
        for split in splits:
            yaml_dict = {"group": f"vizwiz_{task}", "task": f"vizwiz_{task}_{split}", "include": f"_default_template_{task}_yaml", "test_split": split}
            if split == "train":
                yaml_dict.pop("group")
            else:
                dump_tasks.append(f"vizwiz_{task}_{split}")

            save_path = f"./vizwiz_{task}_{split}.yaml"
            print(f"Saving to {save_path}")
            with open(save_path, "w") as f:
                yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    group_dict = {"group": "vizwiz_vqa", "task": dump_tasks}

    with open("./_vizwiz_vqa.yaml", "w") as f:
        yaml.dump(group_dict, f, default_flow_style=False, indent=4)

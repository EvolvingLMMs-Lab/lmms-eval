import os

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))


num_variants = 10

for i in range(num_variants):
    yaml_data = {}
    yaml_data["test_split"] = f"sample_variant{i+1}"
    yaml_data["include"] = "_default_template_yaml"
    yaml_data["task"] = f"dynamath_sample_variant{i+1}"
    with open(os.path.join(current_dir, f"dynamath_sample_variant{i+1}.yaml"), "w") as f:
        yaml.dump(yaml_data, f)

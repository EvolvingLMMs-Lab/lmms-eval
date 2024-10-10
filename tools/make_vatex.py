import json

from datasets import Dataset, load_dataset

with open("data/vatex_public_test_english_v1.1.json", "r") as f:
    data = json.load(f)

for da in data:
    da["url"] = "https://www.youtube.com/watch?v=" + da["videoID"]

vatex_dataset = Dataset.from_list(data)
# vatex_dataset.rename_columns({
#     'videoID': 'video_name',
#     'enCap': 'caption'
# }) #if change name is needed
hub_dataset_path = "lmms-lab/vatex_from_url"

vatex_dataset.push_to_hub(repo_id=hub_dataset_path, split="test", config_name="vatex_test", token=True)

with open("data/vatex_validation_v1.0.json", "r") as f:
    data = json.load(f)
for da in data:
    da["url"] = "https://www.youtube.com/watch?v=" + da["videoID"]

vatex_dataset = Dataset.from_list(data)
# vatex_dataset.rename_columns({
#     'videoID': 'video_name',
#     'enCap': 'caption'
# }) #if change name is needed
hub_dataset_path = "lmms-lab/vatex_from_url"

vatex_dataset.push_to_hub(repo_id=hub_dataset_path, split="validation", config_name="vatex_val_zh", token=True)

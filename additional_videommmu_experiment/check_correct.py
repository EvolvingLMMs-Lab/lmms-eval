import json

# Path to your input JSON file
# input_file = '/mnt/sfs-common/krhu/lmms-eval/logs/20241023_165906_samples_videommmu_application.jsonl'  # replace with your actual JSON file path
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/20241028_091708_samples_application_image.jsonl"
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/20241022_183554_samples_videommmu_comprehension.jsonl"
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/20241030_173258_samples_application_image.jsonl"
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/20241028_004810_samples_videommmu_application.jsonl"
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/lmms-lab__LLaVA-Video-7B-Qwen2/20241031_121246_samples_videommmu_application.jsonl"
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/lmms-lab__LLaVA-Video-7B-Qwen2/20241024_151653_samples_videommmu_application.jsonl"
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/lmms-lab__llava-onevision-qwen2-7b-ov/20241025_113900_samples_videommmu_application.jsonl"
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/lmms-lab__llava-onevision-qwen2-7b-ov/20241031_140745_samples_videommmu_application.jsonl"
# input_file = "logs/lmms-lab__llava-onevision-qwen2-7b-ov/20241101_163246_samples_videommmu_application.jsonl"
# input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/lmms-lab__LLaVA-Video-7B-Qwen2/20241101_153849_samples_videommmu_application.jsonl"
input_file = "/mnt/sfs-common/krhu/lmms-eval/logs/lmms-lab__LLaVA-Video-7B-Qwen2/20241024_164022_samples_videommmu_comprehension.jsonl"
output_file = "comprehension_llavavid.json"

# List to store IDs where "answer" matches "parsed_pred"
correct_ids = []

# Read the JSON file line by line
with open(input_file, "r") as f:
    for line in f:
        entry = json.loads(line.strip())

        # Check if "answer" matches "parsed_pred"
        if entry["mmmu_acc"]["answer"] == entry["mmmu_acc"]["parsed_pred"]:
            correct_ids.append(entry["doc"]["id"])

# Save the matched IDs to the output file
with open(output_file, "w") as f:
    json.dump({"correct_ids": correct_ids}, f, indent=4)

print(f"Matched IDs have been saved to {output_file}")

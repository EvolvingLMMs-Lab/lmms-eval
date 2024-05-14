# from datasets import load_dataset
# sub_dataset_val = load_dataset('MMMU/MMMU', 'Accounting', split='validation', cache_dir='datasets')
# sub_dataset_val = sub_dataset_val.select(range(10))

import json
import os
import numpy as np
from PIL import Image

def generate_placeholder_data(num_entries):
    data = []
    for i in range(1, num_entries + 1):
        entry = {
            "id": str(i),
            "question": "<image 1> ",
            "options": [],
            "explanation": "",
            "img_type": [],
            "answer": "",
            "topic_difficulty": "",
            "question_type": "multiple-choice",
            "subfield": ""
        }
        data.append(entry)
    return data

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def create_image_folders(data):
    for entry in data:
        folder_name = os.path.join(f"M-MMMU/submit/{name}/images/",entry["id"])
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # Create random noise image
        noise_image = np.random.rand(100, 100, 3) * 255
        noise_image = noise_image.astype(np.uint8)
        image_path = os.path.join(folder_name, '<image 1>.jpg')
        
        # Save the noise image
        Image.fromarray(noise_image).save(image_path)

for name in ['akari','anjali','graham','lintang','simran']:
    num_entries = 20  # Change this to the number of entries you want
    filename = f'M-MMMU/submit/{name}/week1.json'
    data = generate_placeholder_data(num_entries)
    write_json(filename, data)
    create_image_folders(data)

print(f"Generated {num_entries} placeholders in {filename} and created corresponding folders.")

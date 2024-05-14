# from datasets import load_dataset
# sub_dataset_val = load_dataset('MMMU/MMMU', 'Accounting', split='validation', cache_dir='datasets')
# sub_dataset_val = sub_dataset_val.select(range(10))

import json
import os
import numpy as np
from PIL import Image

def generate_placeholder_data(num_entries):
    data = []

    # Real example 1 with id = example_1
    example_1 = {
        "id": "example_1",
        "question": "Donna Donie, CFA, has a client who believes the common stock price of TRT Materials (currently $58 per share) could move substantially in either direction in reaction to an expected court decision involving the company. The client currently owns no TRT shares, but asks Donie for advice about implementing a strangle strategy to capitalize on the possible stock price movement. A strangle is a portfolio of a put and a call with different exercise prices but the same expiration date. Donie gathers the TRT option-pricing data: <image 1> Calculate, at expiration for long strangle strategy, the Maximum possible loss per share.",
        "options": "['$9.00', '$5.00', 'the Maximum possible loss is unlimited']",
        "explanation": "",
        "img_type": "['Tables']",
        "answer": "A",
        "topic_difficulty": "Easy",
        "question_type": "multiple-choice",
        "discipline": "Business",
        "subject": "Accounting",
        "subfield": "Investment"
    }
    data.append(example_1)

    # Real example 2 with id = example_2
    example_2 = {
        "id": "example_2",
        "question": "Among the following rests, which one does not have a compound dotted duration?",
        "options": "['<image 1>', '<image 2>', '<image 3>', '<image 4>']",
        "explanation": "",
        "img_type": "['Sheet Music']",
        "answer": "B",
        "topic_difficulty": "Medium",
        "question_type": "multiple-choice",
        "discipline": "Art and Design",
        "subject": "Music",
        "subfield": "Music"
    }
    data.append(example_2)

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
            "discipline": "",
            "subject": "",
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

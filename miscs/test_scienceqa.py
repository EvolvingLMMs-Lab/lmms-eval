from datasets import load_dataset

dataset = load_dataset("Otter-AI/ScienceQA", trust_remote_code=True)["test"]
for doc in dataset:
    print(doc["id"])

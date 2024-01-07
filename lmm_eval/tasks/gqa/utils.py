from datasets import load_dataset
prompt = " Answer the question using a single word or phrase."
raw_image_data = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")

images_dataset = {}
for row in raw_image_data:
    images_dataset[row["id"]] = row["image"].convert("RGB")

def gqa_doc_to_visual(doc):
    image = images_dataset[doc["imageId"]]
    return [image]
    
def gqa_doc_to_text(doc):
    question = doc["question"]
    return f"USER: <image>\n{question}{prompt}\nASSISTANT:"
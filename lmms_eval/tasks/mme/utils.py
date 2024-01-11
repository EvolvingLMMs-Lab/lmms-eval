replace_prompt = "Please answer yes or no."
def mme_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def mme_doc_to_text(doc):
    question = doc["question"]
    question = question.replace(replace_prompt, "").strip()
    return f"{question}\nAnswer the question using a single word or phrase."
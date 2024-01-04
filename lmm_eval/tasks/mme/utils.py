replace_prompt = "Please answer yes or no."
prompt = "\nAnswer the question using a single word or phrase."
def doc_to_visual(doc):
    return [doc["image"]]

def doc_to_text(doc):
    question = doc["question"]
    question = question.replace(replace_prompt, "").strip()
    question = f"{question}{prompt}"
    return f"USER: <image>\n{question}\nASSISTANT:"
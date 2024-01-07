def doc_to_text(doc):
        question, choices = doc["question"], doc["choices"]
        len_choices = len(choices)
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join(
            [f"{option}. {choice}" for option, choice in zip(options, choices)]
        )
        return f"USER: <image>\n{question}\n{choices_str}\nAnswer the question using a single word or phrase.\nASSISTANT:",
    
def doc_to_visual(doc):
    return [doc["image"]]
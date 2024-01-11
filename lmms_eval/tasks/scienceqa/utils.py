def sqa_doc_to_text(doc):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join(
        [f"{option}. {choice}" for option, choice in zip(options, choices)]
    )
    if context:
        context = f"Context: {context}\n"
    return f"{context}{question}\n{choices_str}\nAnswer with the option's letter from the given choices directly."
    
def sqa_doc_to_visual(doc):
    
    return [doc["image"].convert("RGB")]

def sqa_doc_to_target(doc):
    len_choices = len(doc["choices"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["answer"]]
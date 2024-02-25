def doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        return doc["question"]
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs.get("pre_prompt", "")
    post_prompt = model_specific_prompt_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"

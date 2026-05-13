# Utils for visres_bench
def visres_bench_doc_to_text(doc, prompt_kwargs=None):
    """Use question_column in lmms_eval_specific_kwargs to switch guided vs generic question."""
    if prompt_kwargs is None:
        prompt_kwargs = {}
    question_column = prompt_kwargs.get("question_column", "guided_question")
    text = doc.get(question_column)
    pre_prompt = prompt_kwargs.get("pre_prompt", "")
    post_prompt = prompt_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{text}{post_prompt}"


def visres_bench_doc_to_visual(doc):
    if not doc.get("images", None):
        return None
    imgs = doc["images"]
    if isinstance(imgs, list):
        return [img.convert("RGB") for img in imgs]
    return [imgs.convert("RGB")]


def vp_process_results(doc, result):
    answer = doc["answer"]
    result = result[0]
    result = result.split(")")[0]
    if answer == result:
        accuracy = 1
    else:
        accuracy = 0

    return {
        "exact_match": accuracy,
        "submission": {
            "image": doc["id"],
            "answer": result,
        },
    }

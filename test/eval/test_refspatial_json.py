from lmms_eval.tasks.refspatial.utils import refspatial_doc_to_text_json


def test_refspatial_json_uses_roborefer_qwen_prompt():
    prompt = refspatial_doc_to_text_json({"object": "the red cup"})

    assert prompt == ("Locate the red cup in this image and output the point coordinates " "in JSON format.")

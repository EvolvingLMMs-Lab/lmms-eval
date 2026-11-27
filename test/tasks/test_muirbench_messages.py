"""MuirBench chat input places each image at its <image> marker and keeps the option layout.

Many MuirBench options are a bare ``<image>``. The images must follow the marker order, no literal
marker may remain, and the text between them keeps its line breaks so every option stays on its
own line once the chat template renders the images in place.
"""

from PIL import Image

from lmms_eval.tasks.muirbench import utils

PROMPT = {"pre_prompt": "", "post_prompt": "\nAnswer with the option's letter from the given choices directly."}
COLORS = ["red", "green", "blue", "yellow"]


def _doc(n_images=4, options=None):
    return {
        "question": "Which map overlaps the area in <image>?",
        "options": options or ["None of the choices provided", "<image>", "<image>", "<image>"],
        "image_list": [Image.new("RGB", (4, 4), color) for color in COLORS[:n_images]],
    }


def _content(doc):
    (message,) = utils.muir_doc_to_messages(doc, PROMPT)
    return message["content"]


def test_images_follow_marker_order_and_options_keep_their_lines():
    content = _content(_doc())

    images = [item["url"].getpixel((0, 0)) for item in content if item["type"] == "image"]
    assert images == [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0)]
    texts = [item["text"] for item in content if item["type"] == "text"]
    assert all("<image>" not in text for text in texts)
    rendered = "".join("[IMG]" if item["type"] == "image" else item["text"] for item in content)
    assert rendered == ("Which map overlaps the area in [IMG]?\nA. None of the choices provided\nB. [IMG]\nC. [IMG]\nD. [IMG]\nAnswer with the option's letter from the given choices directly.")


def test_text_only_options_keep_the_question_image_in_place():
    content = _content(_doc(n_images=1, options=["a lake", "a river"]))

    assert [item["type"] for item in content] == ["text", "image", "text"]
    assert content[2]["text"] == "?\nA. a lake\nB. a river\nAnswer with the option's letter from the given choices directly."


def test_marker_count_mismatch_keeps_the_visuals_first_layout():
    content = _content(_doc(n_images=2))

    assert [item["type"] for item in content] == ["image", "image", "text"]
    assert content[-1]["text"] == utils.muir_doc_to_text(_doc(n_images=2), PROMPT)

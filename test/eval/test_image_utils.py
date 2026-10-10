"""Tests for ``to_rgb`` and its use in the MathVista / MMMU ``doc_to_visual`` functions.

``Image.convert("RGB")`` drops the alpha channel, so transparent pixels keep the
colour stored under them. In MathVista and MMMU that colour is often black, which
turns black axes, labels and line art on a transparent background into an
all-black image. ``to_rgb`` composites transparency onto white first.

No dataset download, no model inference, no API keys required.
"""

import pytest
from PIL import Image

from lmms_eval.tasks._task_utils.image_utils import to_rgb

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _black_on_transparent(mode: str = "RGBA") -> Image.Image:
    """4x4 fully transparent image (black stored under the alpha) with one opaque black pixel."""
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (0, 0, 0, 255))
    return image.convert(mode) if mode != "RGBA" else image


def test_old_behaviour_loses_the_content():
    # Documents the bug: plain convert("RGB") makes everything black.
    flattened = _black_on_transparent().convert("RGB")
    assert set(flattened.getdata()) == {BLACK}


@pytest.mark.parametrize("mode", ["RGBA", "LA", "PA"])
def test_alpha_modes_are_composited_onto_white(mode):
    result = to_rgb(_black_on_transparent(mode))

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == WHITE
    assert result.getpixel((1, 1)) == BLACK


def test_palette_image_with_transparency_info_is_composited_onto_white():
    image = Image.new("P", (4, 4), 0)
    image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    image.putpixel((1, 1), 1)
    image.info["transparency"] = 0  # palette index 0 is transparent

    result = to_rgb(image)

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == WHITE
    assert result.getpixel((1, 1)) == (255, 0, 0)


def test_partial_alpha_blends_with_white():
    image = Image.new("RGBA", (1, 1), (0, 0, 0, 128))

    value = to_rgb(image).getpixel((0, 0))

    assert value == (127, 127, 127)


def test_opaque_rgba_matches_plain_conversion():
    image = Image.new("RGBA", (3, 3), (10, 20, 30, 255))

    assert list(to_rgb(image).getdata()) == list(image.convert("RGB").getdata())


@pytest.mark.parametrize("mode", ["RGB", "L", "P", "CMYK", "1"])
def test_images_without_transparency_are_unchanged(mode):
    image = Image.effect_noise((8, 8), 64).convert(mode)

    assert to_rgb(image).tobytes() == image.convert("RGB").tobytes()


def test_returns_a_new_image_and_leaves_the_input_untouched():
    image = _black_on_transparent()
    before = image.tobytes()

    result = to_rgb(image)

    assert result is not image
    assert image.mode == "RGBA"
    assert image.tobytes() == before


def test_mmmu_doc_to_visual_composites_transparent_images():
    from lmms_eval.tasks.mmmu import utils

    doc = {
        "question_type": "open",
        "question": "What is the value of <image 1>?",
        "options": "[]",
        "image_1": _black_on_transparent(),
    }

    visual = utils.mmmu_doc_to_visual(doc)

    assert len(visual) == 1
    assert visual[0].mode == "RGB"
    assert visual[0].getpixel((0, 0)) == WHITE
    assert visual[0].getpixel((1, 1)) == BLACK


def test_mathvista_doc_to_visual_composites_transparent_images():
    utils = pytest.importorskip("lmms_eval.tasks.mathvista.utils")

    visual = utils.mathvista_doc_to_visual({"decoded_image": _black_on_transparent()})

    assert len(visual) == 1
    assert visual[0].mode == "RGB"
    assert visual[0].getpixel((0, 0)) == WHITE
    assert visual[0].getpixel((1, 1)) == BLACK

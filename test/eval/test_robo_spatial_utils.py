import numpy as np
from PIL import Image

from lmms_eval.tasks.robo_spatial import utils


def test_save_mask_keys_cache_by_mask_content(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "_MASK_CACHE_DIR", str(tmp_path))
    question = "Place the object to the left."
    first_mask = Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8))
    second_mask = Image.fromarray(np.array([[0, 0], [255, 0]], dtype=np.uint8))

    first = utils._save_mask({"question": question, "mask": first_mask})
    second = utils._save_mask({"question": question, "mask": second_mask})

    assert first["mask_path"] != second["mask_path"]
    np.testing.assert_array_equal(np.array(Image.open(first["mask_path"])), np.array(first_mask))
    np.testing.assert_array_equal(np.array(Image.open(second["mask_path"])), np.array(second_mask))

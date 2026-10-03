"""Exercise real math parser imports and scoring in the CPU contract environment."""

import importlib

import pytest

from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn
from lmms_eval.tasks._task_utils.reasoning_utils import acc_reward
from lmms_eval.tasks.mathvision.eval_utils import eval_tuple


@pytest.mark.parametrize("module_name", ["mathvision.eval_utils", "emma.utils", "stare.utils"])
@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        (r"\frac{1}{2}", "0.5", True),
        ("2+2", "4", True),
        (r"\frac{3}{4}", "0.5", False),
    ],
)
def test_task_latex_equivalence(monkeypatch: pytest.MonkeyPatch, module_name: str, prediction: str, gold: str, expected: bool) -> None:
    """All migrated tasks must accept equivalent numbers and reject wrong ones."""
    # Configure the EMMA judge without credentials; this test never calls it.
    monkeypatch.setenv("API_TYPE", "local")
    module = importlib.import_module(f"lmms_eval.tasks.{module_name}")
    assert module.is_equal(prediction, gold) is expected


@pytest.mark.parametrize(("expression", "expected"), [(r"(2+2,\frac{1}{2})", "(4,0.5)"), (r"[2+2,\frac{1}{2}]", "[4,0.5]")])
def test_mathvision_collection_values(expression: str, expected: str) -> None:
    """Tuple and list answers must retain their arithmetic evaluation behavior."""
    assert eval_tuple(expression) == expected


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [(r"$\frac{1}{2}$", "0.5", 1.0), (r"$\frac{3}{4}$", "0.5", 0.0)],
)
def test_reasoning_math_fallback(prediction: str, gold: str, expected: float) -> None:
    """The reasoning scorer must reach a working math-verify fallback."""
    assert acc_reward(prediction, gold) == expected


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [(r"\boxed{\frac{1}{2}}", "0.5", 1.0), (r"\boxed{\frac{3}{4}}", "0.5", 0.0)],
)
def test_shared_math_verify_configuration(prediction: str, gold: str, expected: float) -> None:
    """The shared normalizer and timeout adapter must work with the new version."""
    score, extracted = MathVerifyFn(silent=False)(prediction, gold)
    assert score == expected
    assert extracted

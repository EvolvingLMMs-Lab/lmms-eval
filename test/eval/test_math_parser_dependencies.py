"""Exercise real math parser imports and scoring in the CPU contract environment."""

import importlib

import pytest

from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn
from lmms_eval.tasks._task_utils.reasoning_utils import acc_reward
from lmms_eval.tasks.mathvision.eval_utils import eval_tuple, is_equal
from lmms_eval.tasks.mathvision.utils import mathvision_process_results


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


@pytest.mark.parametrize(("expression", "expected"), [(r"(2+2,\frac{1}{2})", "(4,0.5)"), (r"[2+2,\frac{1}{2}]", "[4,0.5]"), (r"(\sqrt{3},\pi)", "(1.73,3.14)")])
def test_mathvision_collection_values(expression: str, expected: str) -> None:
    """Tuple and list answers must retain their arithmetic evaluation behavior."""
    assert eval_tuple(expression) == expected


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("3.46", r"2\sqrt{3}", True),
        (r"\sqrt{12}", r"2\sqrt{3}", True),
        ("12.57", r"4\pi", True),
        ("3.47", r"2\sqrt{3}", False),
    ],
)
def test_mathvision_math_names(prediction: str, gold: str, expected: bool) -> None:
    """Like the MATH-V reference, math names such as sqrt and pi must evaluate."""
    assert is_equal(gold, prediction) is expected


@pytest.mark.parametrize(("response", "expected"), [(r"The side is \boxed{3.46}", True), (r"The side is \boxed{3.47}", False)])
def test_mathvision_process_results_math_names(response: str, expected: bool) -> None:
    """The standard MathVision metric must accept a decimal answer for a sqrt gold."""
    result = mathvision_process_results({"answer": r"2\sqrt{3}", "options": []}, [response])
    assert result["mathvision_standard_eval"]["scores"] == [expected]


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

"""Corrected answer extraction, exposed as a separate metric.

`eval_utils.py` is a direct port of the official MATH-V evaluation code
(https://github.com/mathllm/MATH-V/blob/main/evaluation/utils.py) and stays on it
byte for byte, so `mathvision_standard_eval` keeps matching the benchmark's
reference implementation and the numbers published for the paper and the official
leaderboard. The chain below is the same extraction with four corrections, and
only `mathvision_relaxed_eval` uses it:

* `_fix_sqrt` braced the leading `[` of a root index, turning `\\sqrt[3]{8}` into
  `\\sqrt{[}3]{8}` - invalid LaTeX that latex2sympy cannot post-process, so
  `is_equal` fell through to comparing parse failures. (#1506)
* `_strip_string` gated `_fix_fracs` behind `if "sqrt" in string`, so `\\frac12`
  shorthand was never normalized in a sqrt-free answer, and latex2sympy then
  failed to parse it inside `is_equal`. (#1508)
* `find_math_answer` deleted the `\\text`/`\\mbox` command name but kept the
  braces, so `\\boxed{\\text{Sunday}}` became `{sunday}` and scored False against
  the bare gold `Sunday`. (#1510)
* `find_math_answer` matched `oxed{(.*)}` greedily, spanning from the first box to
  the last `}` of the whole string; with a retracted answer the `}`-split
  fallback then resurrected the first box instead of the last. (#1512)

Everything the four corrections do not touch - `is_equal`, `is_number`,
`eval_tuple`, `_fix_fracs`, `_fix_a_slash_b`, `_remove_right_units` - is imported
from `eval_utils`, so the two paths cannot drift apart outside these functions.
The regression tests assert both directions for each correction: the reference
helper keeps reproducing the MATH-V behaviour, and the relaxed one produces the
corrected result.
"""

import re

from lmms_eval.tasks.mathvision.eval_utils import _fix_a_slash_b, _fix_fracs, _remove_right_units


def _fix_sqrt(string):
    # Check if "\sqrt" is not in the string. If not, return the string as is.
    if "\\sqrt" not in string:
        return string

    # Split the string based on the "\sqrt" substring.
    splits = string.split("\\sqrt")

    # The initial portion of the string before the first occurrence of "\sqrt".
    new_string = splits[0]

    # Loop through each split portion (after the initial one).
    for split in splits[1:]:
        # If the split portion is non-empty and the first character isn't a '{',
        # then it means the argument of the sqrt is not enclosed in braces.
        # An optional root index ("\\sqrt[3]{8}") starts with '[' and is already
        # well-formed: bracing its '[' corrupted it to '\\sqrt{[}3]{8}', invalid
        # LaTeX that kills the latex2sympy equivalence path in is_equal.
        if len(split) > 0 and split[0] not in "{[":
            a = split[0]
            # Add braces around the first character and append the rest of the split portion.
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            # If the split portion starts with a '{' or a root index '[', it's already correct.
            new_substr = "\\sqrt" + split
        # Add the new substring to our result string.
        new_string += new_substr

    return new_string


def _strip_string(string):
    # Remove linebreaks
    string = string.replace("\n", "")

    # Remove inverse spaces
    string = string.replace("\\!", "")

    # Replace double backslashes with a single backslash
    string = string.replace("\\\\", "\\")

    # Replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # Remove \left and \right LaTeX commands
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove degree notation
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs (potentially used for inline math in LaTeX)
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # Remove units (assumed to be on the right). Note: The function _remove_right_units is not provided.
    string = _remove_right_units(string)

    # Remove percentage notations
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # Handle floating numbers starting with "."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # If there are equalities or approximations, only consider the value after them
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]

    # Fix sqrt values not wrapped in curly braces. Note: The function _fix_sqrt is not provided.
    if "sqrt" in string:
        string = _fix_sqrt(string)

    # Remove all spaces
    string = string.replace(" ", "")

    # Transform certain fraction notations to the desired format.
    # Unconditional, matching the canonical hendrycks math_equivalence:
    # the "sqrt" guard here was a copy of the _fix_sqrt guard above and
    # made \frac12-style shorthand unreachable for every sqrt-free
    # answer, which latex2sympy then fails to parse inside is_equal.
    string = _fix_fracs(string)

    # Convert 0.5 to its fraction representation
    if string == "0.5":
        string = "\\frac{1}{2}"

    # Fix fractions represented with a slash. Note: The function _fix_a_slash_b is not provided.
    string = _fix_a_slash_b(string)

    return string


def _last_boxed_content(s: str):
    """Return the content of the last `\boxed{...}` span, brace-balanced.

    The previous greedy regex ("oxed{(.*)}", re.S) spanned from the first
    box to the last brace of the whole string; with two boxes ("\boxed{2}.
    wait, \boxed{3}") the }-split fallback below then resurrected the
    FIRST, retracted answer, while this function's own [-1] indexing (and
    lm-eval's last_boxed_only_string convention) intend the last.
    """
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j]
    return None  # unbalanced: fall through to the whole-string fallback


def find_math_answer(s: str) -> str:
    s = s.lower()
    if "{}" in s:
        s = s.replace("{}", "")

    boxed = _last_boxed_content(s)
    ans = boxed if boxed is not None else s  # no (or unbalanced) box: whole string.

    # If there's a closing bracket without an opening bracket before it, consider everything before it.
    if ans.find("}") != -1 and (ans.find("{") == -1 or ans.find("}") < ans.find("{")):
        ans = ans.split("}")[0]

    # Extract the value after the equals sign or approx symbol.
    ans = ans.split("=")[-1]
    ans = ans.split("\\approx")[-1]

    # Clean the string from various LaTeX formatting.
    ans = ans.replace(" ", "").replace("\\,", "").replace("∞", "\\infty")
    ans = ans.replace("+\infty", "\\infty").replace("\\\\", "\\").replace("\n", "")
    # Unwrap \text{...}/\mbox{...} content instead of deleting only the
    # command name: leftover braces corrupt the answer (\text{Sunday} ->
    # "{sunday}", which latex2sympy cannot post-process, so is_equal
    # scores False against the bare gold "Sunday"). Only brace-balanced
    # spans are unwrapped; anything with nested braces falls back to the
    # legacy bare replace, because "{\frac{1}{2}}" parses and must keep
    # doing so.
    ans = re.sub(r"\\(?:text|mbox)\{([^{}]*)\}", r"\1", ans)
    ans = ans.replace("\\text", "").replace("\\mbox", "")
    ans = ans.replace("bmatrix", "pmatrix")
    ans = ans.replace("\\left", "").replace("\\right", "").replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "").replace("{km}", "").replace("km", "")

    return _strip_string(ans)

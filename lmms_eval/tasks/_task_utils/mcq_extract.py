"""Extract MCQ selections from scoped answers and bounded choice tokens.

This is a format parser, not a judge of arbitrary prose. Explicit answers take
precedence over implicit labels, and ambiguous selections return an empty string.
Benchmark-specific official scorers and reasoning stripping remain separate.
"""

import re
import unicodedata
from typing import List, NamedTuple, Optional, Set

_DEFAULT_CHOICES = list("ABCDEFGH")

# Keep containers separate from declarations: closing tags are never candidates.
# The bare ``answer>`` marker and unclosed opening tags preserve legacy inputs.
_CONTAINER_START = re.compile(r"<answer\b[^<>]*>|(?<![\w</])answer>|\\boxed\s*\{", re.IGNORECASE)
_ANSWER_OPEN = re.compile(r"<answer\b", re.IGNORECASE)
_ANSWER_MARKUP = re.compile(r"<answer\b[^<>]*>|(?<![\w</])answer>|</answer\s*>", re.IGNORECASE)
_ANSWER_CLOSE = re.compile(r"</answer\s*>", re.IGNORECASE)
_DECLARATION = re.compile(
    r"(?P<prefix>"
    r"(?<!\w)(?:(?:the|my)\s+)?(?:(?:correct|best|final)\s+)?(?:answer|option|choice)\s*(?:is\b|[:：=])"
    r"|(?<!\w)i\s+(?:choose|select|pick)\b"
    r"|答案是|答案为|选|옵션|정답은|답은|답\s*[:：]|答えは"
    r")"
    r"|(?P<postfix>(?<!\w)option\s+(?P<option>[a-z])(?!\w)\s+is\s+(?:the\s+)?(?:correct|right|best)\b)",
    re.IGNORECASE,
)
_OPTION_PREFIX = re.compile(r"(?:option|choice)\b\s*[:：]?\s*", re.IGNORECASE)
_LATEX_WRAPPER = re.compile(r"\\(?:text|mathrm|mathbf|textbf)\s*\{")
_PAREN_CHOICE = re.compile(r"(?<!\w)(?:\(\s*([A-Za-z])\s*\)|（\s*([A-Za-z])\s*）)(?!\w)")
_OPTION_LABEL = re.compile(r"(?<!\w)([A-Za-z])[.):](?=\s|$)")
_ASSERTION = re.compile(r"(?:because\b|(?:is|seems|looks|appears)\s+(?:(?:the|a)\s+)?(?:correct|right|best|answer|option|choice)\b)", re.IGNORECASE)
_ALTERNATIVE = re.compile(r"(?:and\s*/\s*or\b|and\b|or\b|[/&,+|]|或(?:者)?|和|또는|か|と)\s*", re.IGNORECASE)
_IMPLICIT_ALTERNATIVES = re.compile(r"(?<!\w)[A-Z]\s*(?:[,/&+|]|and\b|or\b)\s*[A-Z](?!\w)")
_URL = re.compile(r"(?<!\w)(?:[a-z][a-z0-9+.-]*://|www\.)\S*", re.IGNORECASE)
_TRAILING_SEGMENT = re.compile(r"(?:^|[\n,;，；])\s*([^\n,;，；]+)\s*$")
_WRAPPERS = (("**", "**"), ("__", "__"), ("`", "`"), ("*", "*"), ("_", "_"), ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("(", ")"), ("（", "）"), ("[", "]"), ("{", "}"), ("$", "$"))
_PUNCTUATION = " \t\r\n.!?,;:。！？，；："


class _Choice(NamedTuple):
    letter: str
    tail: str
    wrapped: bool
    uppercase: bool


def _word_character(character: str) -> bool:
    """Include Unicode combining marks and joins in word/identifier boundaries."""
    return character.isalnum() or character == "_" or unicodedata.category(character) in {"Mn", "Mc", "Me", "Cf"}


def _read_choice(fragment: str, grammatical_suffix: str = "") -> Optional[_Choice]:
    """Read one adjacent ASCII choice, with paired presentation wrappers."""
    text = fragment.lstrip().lstrip(":：=,，;；").lstrip()
    closers = []
    option_prefix_used = False
    while text:
        wrapper = _LATEX_WRAPPER.match(text)
        if wrapper:
            closers.append("}")
            text = text[wrapper.end() :].lstrip()
            continue
        prefix = _OPTION_PREFIX.match(text) if not option_prefix_used else None
        if prefix:
            option_prefix_used = True
            text = text[prefix.end() :].lstrip()
            continue
        pair = next((pair for pair in _WRAPPERS if text.startswith(pair[0])), None)
        if pair is None:
            break
        closers.append(pair[1])
        text = text[len(pair[0]) :].lstrip()

    if not text or not text[0].isascii() or not text[0].isalpha():
        return None
    letter, tail = text[0], text[1:]
    # Python's Unicode word boundary also excludes letters inside Chinese text,
    # identifiers and accented words; ASCII-only regex boundaries would not.
    if grammatical_suffix and tail.startswith(grammatical_suffix):
        tail = tail[len(grammatical_suffix) :]
    if tail and _word_character(tail[0]) and not (closers and tail.startswith(closers[-1])):
        return None
    punctuation = ""
    for closer in reversed(closers):
        tail = tail.lstrip()
        # Punctuation may be inside a presentation wrapper: ``**B.**``.
        inner_punctuation = ".:：!?。！？" + (")" if closer != ")" else "")
        punctuation_end = len(tail) - len(tail.lstrip(inner_punctuation))
        punctuation += tail[:punctuation_end]
        tail = tail[punctuation_end:].lstrip()
        if not tail.startswith(closer):
            return None
        tail = tail[len(closer) :]
    tail = punctuation + tail
    if tail and _word_character(tail[0]):
        return None
    if tail and tail[0] in ".:-/'’" and len(tail) > 1 and (tail[1].isalnum() or tail[1] in "/_"):
        # ``B.com``, ``C:/path``, ``A-list`` and ``B's`` are not labels.
        # A slash followed by a single choice is handled as an alternative below.
        if tail[0] != "/":
            return None
    return _Choice(letter.upper(), tail, bool(closers), letter.isupper())


def _has_alternative(choice: _Choice) -> bool:
    """Reject an adjacent second selection, not letters in an explanation."""
    tail = choice.tail.lstrip(" \t")
    label_punctuation = tail[:1] in (".", ":", ")", "：")
    if label_punctuation:
        tail = tail[1:].lstrip(" \t")
    connector = _ALTERNATIVE.match(tail)
    if connector:
        return tail.startswith("/") or _read_choice(tail[connector.end() :]) is not None
    # A compact sequence ``B C`` is ambiguous. A separate later implicit line
    # does not invalidate an earlier explicit declaration.
    if label_punctuation or tail.startswith(("\n", "\r")):
        return False
    other = _read_choice(tail)
    return other is not None and not other.tail.strip(_PUNCTUATION)


def _declared_answer(text: str, allowed: Set[str]) -> Optional[str]:
    """Return the last declaration, preserving an invalid declaration as empty."""
    answer = None
    for declaration in _DECLARATION.finditer(text):
        fragment = declaration.group("option") + text[declaration.end() :] if declaration.group("postfix") else text[declaration.end() :]
        marker = declaration.group("prefix") or ""
        suffix = "입니다" if marker.startswith(("옵션", "정답은", "답은", "답")) else "です" if marker == "答えは" else ""
        choice = _read_choice(fragment, grammatical_suffix=suffix)
        answer = choice.letter if choice and choice.letter in allowed and not _has_alternative(choice) else ""
    return answer


def _closing_brace(text: str, start: int) -> Optional[int]:
    """Find a box's closing brace, allowing balanced presentation wrappers."""
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _last_container(text: str) -> Optional[str]:
    """Scope to the last container; malformed final evidence never falls back."""
    tag_open = False
    for tag in _ANSWER_MARKUP.finditer(text):
        if tag.group().startswith("</"):
            if not tag_open:
                return ""
            tag_open = False
        else:
            if tag_open:
                return ""
            tag_open = True
    starts = list(_CONTAINER_START.finditer(text))
    raw_tags = list(_ANSWER_OPEN.finditer(text))
    tag_starts = {opening.start() for opening in starts if opening.group().startswith("<")}
    if any(tag.start() not in tag_starts for tag in raw_tags):
        return ""
    if not starts:
        return None
    box_end = -1
    for opening in starts:
        if not opening.group().startswith("\\"):
            continue
        if opening.start() < box_end:
            return ""
        closing_brace = _closing_brace(text, opening.end())
        box_end = closing_brace if closing_brace is not None else len(text)
    start = starts[-1]
    if not start.group().startswith("\\"):
        closing = _ANSWER_CLOSE.search(text, start.end())
        return text[start.end() : closing.start() if closing else len(text)]
    # An unfinished LaTeX container must not expose earlier answer candidates.
    return text[start.end() : box_end] if box_end < len(text) else ""


def _implicit_answer(text: str, allowed: Set[str]) -> str:
    """Accept structural labels while abstaining on conflicting option lists."""
    urls = [match.span() for match in _URL.finditer(text)]
    matches = [
        match
        for pattern in (_PAREN_CHOICE, _OPTION_LABEL)
        for match in pattern.finditer(text)
        if (match.start() == 0 or not _word_character(text[match.start() - 1])) and (match.end() == len(text) or not _word_character(text[match.end()])) and not any(start <= match.start() < end for start, end in urls)
    ]
    labels = {next(group for group in match.groups() if group).upper() for match in matches}
    for line in text.splitlines():
        choice = _read_choice(line)
        if choice and choice.uppercase and not choice.tail.strip(_PUNCTUATION):
            labels.add(choice.letter)
    candidates = set()
    for match in matches:
        if match.re is not _PAREN_CHOICE:
            continue
        choice = _read_choice(text[match.start() :])
        if choice and _has_alternative(choice):
            return ""
        candidates.add(next(group for group in match.groups() if group).upper())

    leading = _read_choice(text)
    if leading and _has_alternative(leading):
        return ""
    if leading:
        tail = leading.tail.lstrip()
        if leading.wrapped or not tail.strip(_PUNCTUATION) or tail.startswith((".", ":", ")", "：")) or _ASSERTION.match(tail):
            candidates.add(leading.letter)
    # Unscoped comma-separated alternatives must not become a trailing answer.
    # A leading selection may justify itself by discussing other choices.
    if not (leading and leading.letter in candidates) and any(not any(start <= match.start() < end for start, end in urls) for match in _IMPLICIT_ALTERNATIVES.finditer(text)):
        return ""

    trailing = _TRAILING_SEGMENT.search(text)
    if trailing and not any(start <= trailing.start(1) < end for start, end in urls):
        choice = _read_choice(trailing.group(1))
        if choice and choice.uppercase and not choice.tail.strip(_PUNCTUATION):
            candidates.add(choice.letter)

    if len(labels | candidates) != 1 or not candidates:
        return ""
    return next(iter(candidates)) if candidates <= allowed else ""


def extract_mcq_answer(response: str, choices: Optional[List[str]] = None) -> str:
    r"""Extract a single answer choice, or abstain with ``""``.

    The last ``<answer>``/``\boxed{}`` container scopes parsing to its content.
    Otherwise the last explicit declaration wins, including invalid final
    declarations: ``Answer: B. Final answer: Z`` does not fall back to B when Z
    is unavailable. Adjacent alternatives such as ``A or B`` return empty.

    Without explicit evidence, accept bare letters, leading punctuated labels,
    standalone parenthesized choices and uppercase final lines/comma-separated
    conclusions. Conflicting implicit labels abstain. Ordinary prose and
    embedded letters are not searched for fallback candidates. Markdown and
    paired presentation wrappers preserve choices; lowercase choices require
    an answer format rather than a mention in prose.

    This deliberately conservative format policy cannot resolve arbitrary
    author intent. Callers must strip reasoning blocks before extraction and
    retain any benchmark-specific official scorer.

    Args:
        response: Model output, with reasoning already stripped upstream.
        choices: Single ASCII letters, case insensitive. None or an empty list
            defaults to A through H. Custom alphabets such as I through N work.

    Returns:
        An uppercase offered choice, or an empty string for absent, invalid,
        or ambiguous answer evidence.
    """
    if not response or not response.strip():
        return ""
    allowed = {choice.upper() for choice in choices or _DEFAULT_CHOICES if len(choice) == 1 and choice.isascii() and choice.isalpha()}
    text = response.strip()
    scope = _last_container(text)
    if scope is not None:
        if re.search(r"</?\w", scope):
            return ""
        declared = _declared_answer(scope, allowed)
        if declared is not None:
            return declared
        choice = _read_choice(scope)
        standalone = set()
        for line in scope.splitlines():
            line_choice = _read_choice(line)
            if line_choice and not line_choice.tail.strip(_PUNCTUATION):
                standalone.add(line_choice.letter)
        if choice and len(standalone | {choice.letter}) > 1:
            return ""
        return choice.letter if choice and choice.letter in allowed and not _has_alternative(choice) else ""
    declared = _declared_answer(text, allowed)
    return declared if declared is not None else _implicit_answer(text, allowed)

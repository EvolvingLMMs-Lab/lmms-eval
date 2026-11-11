r"""Read the final yes/no answer of a response, or abstain.

Yes/no scorers read the answer with a short heuristic: the first characters,
the first or last ``yes``/``no`` substring, or an exact match. These read
``eyes``, ``know`` and ``nothing``, miss ``**Yes**`` and ``Yes.``, and read the
words a reasoning chain uses on its way. ``extract_yes_no_answer`` reads only
statements of an answer, at the positions where a response states it, and
returns ``""`` when it is not sure. A scorer calls it first and keeps its own
heuristic as the fallback on ``""``, so a response that the scorer already
reads is scored as before unless the response states a final answer here.

Contract:

* The result is ``"yes"``, ``"no"`` or ``""``. Case is ignored. Presentation
  wrappers around the word are ignored: ``**Yes**``, ``*no*``, backticks,
  quotes, brackets and ``\text{Yes}``.
* A statement of an answer is one of:

  - S1, a container: the last ``<answer>...</answer>`` or ``\boxed{...}``.
    Its content must be yes/no alone, or open with yes/no and then
    punctuation (``\boxed{Yes, the shop is open}``). A last container with
    other content (``\boxed{false}``, ``\boxed{}``) gives ``""``, and nothing
    else in the response is read. A container that the text after it
    contradicts (a declaration, a yes/no line or a paragraph opening with the
    other answer: ``Maybe \boxed{Yes}? ... So the answer is no.``) gives ``""``.
  - S2, a declaration with yes/no directly after it: ``Answer: Yes``,
    ``the answer is no``, ``**Final Answer:** yes``, ``I will answer yes``.
    In a block the last declaration decides. A declaration that does not name yes/no alone
    (``the answer is not yes``, ``the answer is yes or no``, ``the answer is
    no dog``, ``the answer is clear``) or that is hedged in its sentence
    (``perhaps``, ``maybe``, ``if``, ``whether``, ...) makes the block abstain.
  - S3, a line that is only yes/no (with wrappers and a full stop).
  - S4, a block that opens with yes/no followed by the end of the line or by
    punctuation: ``Yes, there is a dog.``, ``No. The table ...``. Not
    ``No dog is visible``, ``No one``, ``Yes or no``, ``Yes/No``, ``Yes?``,
    and not ``No. 12`` (the abbreviation of "number"), also in a container.

* Positions. A container anywhere decides. With several paragraphs (separated
  by a blank line), only the last and the first paragraph are read: the last
  one holds the conclusion of a reasoning chain, the first one an answer that
  is explained afterwards, and it must open with its statement (S2 or S4).
  Different answers at the two ends, or an unusable declaration in the last
  paragraph, give ``""``. In a single paragraph: S2 decides; otherwise the
  opening statement (S4) and the S3 lines must agree.
* The reader never reads yes/no inside prose (``there is no dog``, ``I know``,
  ``cannot``, ``nothing``, ``eyes``) and never infers an answer (``false``,
  ``not visible``, ``there is none`` give ``""``). It reads English only;
  scorers that accept synonyms (true/false, Chinese, y/n) keep them in their
  fallback.
* Input that is not a string, or is empty, gives ``""``.

All patterns are linear in the length of the input, so a degenerate response
of several hundred thousand characters is read in well under a second.
"""

import re
from typing import Optional, Tuple

_ANSWERS = ("yes", "no")

# Characters that can open or close a presentation wrapper around the word.
_OPENERS = " \t\r\n*_`#>\"'“‘([{"
_CLOSERS = " \t*_`\"'”’)]}"
# Punctuation that may follow the word in a statement. ``?`` turns it into a question.
_ENDINGS = ".,;:!()[]-—–。，；：！"
_LATEX_OPEN = re.compile(r"\\(?:text|textbf|textrm|mathrm|mathbf|bf)\s*\{")
_WORD = re.compile(r"(yes|no)\b", re.IGNORECASE)
_NUMBER_SIGN = re.compile(r"\.?\s?\d")

_CONTAINER = re.compile(r"<answer\s*>|\\boxed\s*\{", re.IGNORECASE)
_ANSWER_CLOSE = re.compile(r"</answer\s*>", re.IGNORECASE)
_BLANK_LINE = re.compile(r"\n\s*\n")
_BRACE = re.compile(r"[{}]")
_DECLARATION = re.compile(
    r"(?<!\w)(?:(?:the|my|final|correct|short|right)\s+){0,3}answer"
    r"(?:\s+to\s+(?:the|this)\s+question)?[*_]*\s*(?:is\b\s*:?|[:：=])"
    r"|(?<!\w)i(?:\s+will|['’]ll)\s+(?:answer|say|go\s+with)(?!\w)\s*:?",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"(?<!\w)(?:perhaps|maybe|might|possibly|probably|likely|if|whether|unless|suppose|supposing|assume|assuming|could|would|should)(?!\w)",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?\n]")
_STANDALONE_MAX = 80


def _opening_word(text: str) -> Optional[str]:
    """Read yes/no at the start of ``text``.

    Returns the word when it is a statement (followed by the end of the line or
    by punctuation), ``""`` when the text starts with yes/no that is not one
    (``No dog``, ``Yes or no``, ``Yes/No``, ``Yes?``), and ``None`` when the text
    does not start with yes/no.
    """
    text = text.lstrip(_OPENERS)
    latex = _LATEX_OPEN.match(text)
    if latex:
        text = text[latex.end() :].lstrip(_OPENERS)
    word = _WORD.match(text)
    if word is None:
        return None
    rest = text[word.end() : word.end() + 64]
    if rest[:1] in ("/", "-", "'", "’") and rest[1:2].strip():
        return ""  # Yes/No, No-one, no's
    if _NUMBER_SIGN.match(rest):
        return ""  # "No. 12A-G, Jalan ..." abbreviates "number"
    rest = rest.lstrip(_CLOSERS)
    if not rest or rest[0] in "\r\n" or rest[0] in _ENDINGS:
        return word.group(1).lower()
    return ""


def _standalone(line: str) -> Optional[str]:
    """Return yes/no for a line that holds nothing else, else ``None``."""
    if len(line) > _STANDALONE_MAX:
        return None
    core = line.strip(_OPENERS + _CLOSERS + ".!。")
    latex = _LATEX_OPEN.match(core)
    if latex:
        core = core[latex.end() :].strip(_OPENERS + _CLOSERS + ".!。")
    core = core.lower()
    return core if core in _ANSWERS else None


def _declared(block: str, declaration: "re.Match[str]") -> str:
    """Return the answer a declaration names, or ``""`` when it is unusable."""
    word = _opening_word(block[declaration.end() : declaration.end() + 200])
    if not word:
        return ""
    sentence = block[max(0, declaration.start() - 300) : declaration.start()]
    ends = list(_SENTENCE_END.finditer(sentence))
    if ends:
        sentence = sentence[ends[-1].end() :]
    return "" if _HEDGE.search(sentence) else word


def _stated(block: str, opening: bool = False) -> Optional[str]:
    """Return the answer a block states, ``""`` to abstain, ``None`` when it states nothing.

    With ``opening``, the block must open with its statement.
    """
    start = len(block) - len(block.lstrip(_OPENERS))
    if opening:
        first = _DECLARATION.match(block, start)
        lead = _declared(block, first) if first else _opening_word(block)
        if not lead:
            return None
        last = None
        for last in _DECLARATION.finditer(block):
            pass
        if last is not None and last.start() != start and _declared(block, last) != lead:
            return None  # the paragraph goes on to declare something else
        return lead
    last = None
    for last in _DECLARATION.finditer(block):
        pass
    if last is not None:
        return _declared(block, last)
    stated = {_standalone(line) for line in block.splitlines()}
    stated.add(_opening_word(block))
    stated -= {None, ""}
    if len(stated) == 1:
        return stated.pop()
    return "" if stated else None


def _last_container(text: str) -> Optional[Tuple[str, str]]:
    """Return the content of the last ``<answer>``/``\\boxed{}`` container and the text after it, or ``None``."""
    last = None
    for last in _CONTAINER.finditer(text):
        pass
    if last is None:
        return None
    if last.group().startswith("<"):
        close = _ANSWER_CLOSE.search(text, last.end())
        if close is None:
            return text[last.end() :], ""
        return text[last.end() : close.start()], text[close.end() :]
    depth = 1
    for brace in _BRACE.finditer(text, last.end()):
        depth += 1 if brace.group() == "{" else -1
        if depth == 0:
            return text[last.end() : brace.start()], text[brace.end() :]
    return text[last.end() :], ""


def _states_other(text: str, answer: str) -> bool:
    """Whether ``text`` states the other answer: in a declaration, a yes/no line or a paragraph opening."""
    other = "no" if answer == "yes" else "yes"
    if any(_declared(text, declaration) == other for declaration in _DECLARATION.finditer(text)):
        return True
    if any(_standalone(line) == other for line in text.splitlines()):
        return True
    return any(_opening_word(block) == other for block in _BLANK_LINE.split(text))


def _container_answer(content: str) -> str:
    standalone = _standalone(content.strip())
    if standalone:
        return standalone
    content = content.strip()
    declaration = _DECLARATION.match(content, len(content) - len(content.lstrip(_OPENERS)))
    if declaration:
        return _declared(content, declaration)
    return _opening_word(content) or ""


def extract_yes_no_answer(response) -> str:
    """Return the final ``"yes"``/``"no"`` a response states, or ``""`` (see the module docstring)."""
    if not isinstance(response, str) or not response.strip():
        return ""
    text = response.strip()
    container = _last_container(text)
    if container is not None:
        content, after = container
        answer = _container_answer(content)
        # "\boxed{Yes}? ... So the answer is No." -- a later statement of the other answer: not sure.
        return "" if answer and _states_other(after, answer) else answer
    paragraphs = [block for block in _BLANK_LINE.split(text) if block.strip()]
    if len(paragraphs) > 1:
        last = _stated(paragraphs[-1])
        first = _stated(paragraphs[0], opening=True)
        if last == "" or (first and last and first != last):
            return ""
        return last or first or ""
    return _stated(text) or ""

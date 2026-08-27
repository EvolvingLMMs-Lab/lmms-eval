"""MDPBench evaluation utilities for lmms-eval.

MDPBench (Multilingual Document Parsing Benchmark) evaluates multilingual
document-image-to-Markdown conversion across 17 languages under both
digital-scan and photographed conditions.

Evaluation pipeline (per page):
  1. Model converts a document page image to Markdown.
  2. Prediction is parsed into typed segments: text, formulas, tables
     (ported from MDPBench utils/extract.py::md_tex_filter).
  3. GT elements are extracted from layout_dets via _get_page_elements(),
     a faithful replica of MDPBench End2EndDataset.get_page_elements()
     including its list-append truncated-merge algorithm.
  4. GT text+formula pool is matched against pred text+formula pool jointly
     with the reference quick matcher and its 30-second simple-match fallback.
  5. After matching, formula elements matched to non-formula predictions are
     reclassified into the text pool (_reclassify_formulas).
  6. Ignore categories (header/footer/captions/…) participate in matching
     to absorb corresponding predictions, but are excluded from scoring.
  7. Tables are matched separately; format (HTML vs LaTeX) chosen by count.
  8. Scoring: text edit distance, formula CDM, table TEDS.
  9. Per-page overall = mean(1-text_ed, formula_cdm, table_teds).
 10. Aggregation with language / digital-photo split filters.

Design notes
  * _get_page_elements replicates the original list-append algorithm verbatim,
    including the duplicate-ID behavior in multi-hop truncated chains, so that
    scores are directly comparable with official MDPBench numbers.
  * CDM is a hard required dependency (lazy import, raises ImportError with
    install instructions on first use).
  * TEDS uses the original MDPBench implementation verbatim.

References:
  MDPBench — https://github.com/Delores-Lin/MDPBench
"""

from __future__ import annotations

import atexit
import io
import json
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple

import Levenshtein
from loguru import logger as eval_logger
from PIL import Image

# ---------------------------------------------------------------------------
# Hard dependency: CDM (Character Detection Metric)
# ---------------------------------------------------------------------------
# CDM code is bundled in lmms_eval/tasks/mdpbench/cdm_metric.py
# System requirements : pdflatex, Node.js with KaTeX installed

# Module-level singletons to avoid re-creating expensive objects per sample.
_TEDS_SCORER = None
_CDM_EVALUATOR_INSTANCE = None


def _get_teds_scorer():
    """Return a module-level TEDS singleton (avoids re-import + re-init per table)."""
    global _TEDS_SCORER
    if _TEDS_SCORER is None:
        from lmms_eval.tasks.mdpbench.teds_metric import TEDS

        _TEDS_SCORER = TEDS(structure_only=False, n_jobs=1)
    return _TEDS_SCORER


def _get_cdm_evaluator(output_root=None):
    """Return a module-level CDM singleton with a persistent per-process temp dir."""
    global _CDM_EVALUATOR_INSTANCE
    if _CDM_EVALUATOR_INSTANCE is None:
        try:
            from lmms_eval.tasks.mdpbench.cdm_metric import CDM
        except ImportError as e:
            raise ImportError(
                "Failed to import CDM (required for MDPBench formula evaluation).\n"
                "Make sure the following system dependencies are installed:\n"
                "  - xelatex  (sudo apt-get install texlive-full)\n"
                "  - ImageMagick 7  (magick command)\n"
                "  - Node.js\n"
                "  - scikit-image  (pip install scikit-image)\n"
                f"Original error: {e}"
            ) from e
        # Use a fixed per-process directory so the singleton output_root never changes.
        cdm_dir = output_root or os.path.join(tempfile.gettempdir(), f"mdpbench_cdm_{os.getpid()}")
        os.makedirs(cdm_dir, exist_ok=True)
        _CDM_EVALUATOR_INSTANCE = CDM(output_root=cdm_dir)
        # Clean up on exit
        import shutil

        atexit.register(shutil.rmtree, cdm_dir, True)
    return _CDM_EVALUATOR_INSTANCE


# ---------------------------------------------------------------------------
# Prompt  (from MDPBench scripts/batch_process_*.py)
# ---------------------------------------------------------------------------

_PROMPT = (
    "\nYou are an advanced hybrid OCR engine capable of processing multilingual text mixed "
    "with mathematical notation. Your goal is to transcribe the content with high fidelity. "
    "Strict Rules:\n\n"
    "1. Multilingual Precision: Transcribe text exactly as it appears in the original language. "
    "Do not translate, summarize, or correct original spelling errors.\n\n"
    "2. Math Formatting: Identify all mathematical expressions and convert them into LaTeX.\n\n"
    "3. Inline Math: Use single dollar signs ($x$) for inline math (formulas within a sentence).\n\n"
    "4. Display Math: Use double dollar signs ($$x$$) for display math "
    "(standalone formulas on their own lines).\n\n"
    "5. Layout & Structure: Use Markdown to preserve the visual structure (headers, paragraphs, lists).\n\n"
    "6. Table Formatting: Use HTML tags (e.g., <table>, <tr>, <th>, <td>) "
    "to generate any tables found in the text.\n\n"
    "7. Output Only: Output the transcribed text directly without any conversational filler.\n"
)

# ---------------------------------------------------------------------------
# Category and language constants
# ---------------------------------------------------------------------------

# All GT text categories that enter the mixed matching pool
_GT_MIX_CATEGORIES: List[str] = [
    "text_block",
    "title",
    "code_txt",
    "code_txt_caption",
    "reference",
    "equation_caption",
    "figure_caption",
    "figure_footnote",
    "table_caption",
    "table_footnote",
    "code_algorithm",
    "code_algorithm_caption",
    "header",
    "footer",
    "page_footnote",
    "page_number",
    "equation_isolated",
]
# Subset matched but NOT penalised if missed
_IGNORE_SCORE_CATEGORIES = frozenset(
    {
        "figure_caption",
        "figure_footnote",
        "table_caption",
        "table_footnote",
        "code_algorithm",
        "code_algorithm_caption",
        "header",
        "footer",
        "page_footnote",
        "page_number",
        "equation_caption",
    }
)
_FORMULA_CATEGORIES = frozenset({"equation_isolated"})
_TABLE_CATEGORIES = frozenset({"table"})

_LANGUAGE_MAP: Dict[str, str] = {
    "simplified chinese": "ZH",
    "traditional chinese": "ZH-T",
    "english": "EN",
    "arabic": "AR",
    "german": "DE",
    "spanish": "ES",
    "french": "FR",
    "hindi": "HI",
    "indonesian": "ID",
    "italian": "IT",
    "japanese": "JP",
    "korean": "KO",
    "portuguese": "PT",
    "russian": "RU",
    "thai": "TH",
    "vietnamese": "VI",
    "dutch": "NL",
}
_FILENAME_LANG_MAP: Dict[str, str] = {
    "zh": "ZH",
    "zh-cht": "ZH-T",
    "en": "EN",
    "ar": "AR",
    "de": "DE",
    "es": "ES",
    "fr": "FR",
    "hi": "HI",
    "id": "ID",
    "it": "IT",
    "jp": "JP",
    "ko": "KO",
    "pt": "PT",
    "ru": "RU",
    "th": "TH",
    "vi": "VI",
    "nl": "NL",
}
_LATIN_LANGS = frozenset({"EN", "DE", "ES", "FR", "ID", "IT", "NL", "PT", "VI"})
_NON_LATIN_LANGS = frozenset({"AR", "HI", "JP", "KO", "RU", "TH", "ZH", "ZH-T"})

_MATCH_COST_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# doc_to_* interface
# ---------------------------------------------------------------------------


def mdpbench_doc_to_visual(doc) -> List[Image.Image]:
    for key in ("image", "page_image", "document_image"):
        raw = doc.get(key)
        if raw is None:
            continue
        if isinstance(raw, Image.Image):
            return [raw.convert("RGB")]
        if isinstance(raw, bytes):
            return [Image.open(io.BytesIO(raw)).convert("RGB")]
        if isinstance(raw, str):
            import base64

            return [Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")]
    return []


def mdpbench_doc_to_text(doc, lmms_eval_specific_kwargs=None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{_PROMPT}{kwargs.get('post_prompt', '')}"


def mdpbench_doc_to_messages(doc, lmms_eval_specific_kwargs=None) -> List[Dict]:
    """Build the official image-then-prompt user turn without a system turn."""
    content = [{"type": "image", "url": image} for image in mdpbench_doc_to_visual(doc)]
    content.append(
        {
            "type": "text",
            "text": mdpbench_doc_to_text(doc, lmms_eval_specific_kwargs),
        }
    )
    return [{"role": "user", "content": content}]


def mdpbench_doc_to_target(doc) -> str:
    return doc.get("answer", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_answer(doc) -> dict:
    raw = doc.get("answer", "{}")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}


def _detect_language(doc) -> str:
    answer = _load_answer(doc)
    lang_str = answer.get("page_info", {}).get("page_attribute", {}).get("language", "").lower().strip()
    if lang_str in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[lang_str]
    img_path = answer.get("page_info", {}).get("image_path", "")
    prefix = img_path.split("_")[0].lower() if img_path else ""
    return _FILENAME_LANG_MAP.get(prefix, "UNK")


def _is_digital(doc) -> bool:
    """filename with exactly 3 underscore-parts → digital scan."""
    answer = _load_answer(doc)
    img_path = answer.get("page_info", {}).get("image_path", "")
    stem = img_path.rsplit(".", 1)[0]
    parts = [p for p in stem.split("_") if p]
    return len(parts) == 3


def _get_image_id(doc) -> str:
    answer = _load_answer(doc)
    img_path = answer.get("page_info", {}).get("image_path", "")
    return img_path.rsplit(".", 1)[0] if img_path else "unknown"


# ---------------------------------------------------------------------------
# GT extraction — faithful replica of MDPBench
# End2EndDataset.get_page_elements()
# ---------------------------------------------------------------------------


def _get_page_elements(answer: dict) -> Dict[str, List[Dict]]:
    """Extract GT elements from layout_dets, merging truncated element groups.

    This is a line-by-line port of MDPBench End2EndDataset.get_page_elements()
    (dataset/end2end_dataset.py L53-92), preserving the original list-append
    algorithm verbatim — including the duplicate-ID behavior that occurs in
    multi-hop truncated chains (e.g. A-B + B-C produces merge list [A,B,B,C],
    causing B's text to be concatenated twice).  This is intentional so that
    computed scores are directly comparable with official MDPBench results.

    Returns a dict {category_type: [item_dict, ...]}.
    """
    layout_dets = answer.get("layout_dets", [])
    relations = answer.get("extra", {}).get("relation", [])

    saved_element_dict: Dict[str, List[Dict]] = defaultdict(list)
    related_truncated: List[List[int]] = []  # list of merge groups (anno_id lists)
    truncated_all: Dict[int, any] = {}  # anno_id → "" initially, then → item

    # --- Phase 1: build merge groups (original list-append algorithm) ---
    for relation in relations:
        if relation.get("relation_type") != "truncated":
            continue
        src = relation["source_anno_id"]
        tgt = relation["target_anno_id"]
        truncated_all[src] = ""
        truncated_all[tgt] = ""
        exist_flag = False
        for merge_list in related_truncated:
            if src in merge_list or tgt in merge_list:
                # Append both IDs regardless — may introduce duplicates in
                # multi-hop chains, matching original MDPBench behaviour.
                merge_list.append(src)
                merge_list.append(tgt)
                exist_flag = True
        if not exist_flag:
            related_truncated.append([src, tgt])

    # --- Phase 2: route items into direct dict or truncated holding area ---
    for item in layout_dets:
        if item["anno_id"] not in truncated_all:
            saved_element_dict[item["category_type"]].append(item)
        else:
            truncated_all[item["anno_id"]] = item

    # --- Phase 3: merge truncated groups and add to dict ---
    for merge_list in related_truncated:
        # Collect items; skip IDs still mapped to "" (not found in layout_dets)
        text_block_list = [truncated_all[key] for key in merge_list if truncated_all.get(key) not in ("", None)]
        if not text_block_list:
            continue
        sorted_block = sorted(text_block_list, key=lambda x: x.get("order", 0))
        # Original always reads .text (works because truncated rels are text-only
        # in practice; replicating faithfully here).
        text = ""
        for block in sorted_block:
            text += block.get("text") or ""
        merged_block = {
            "category_type": sorted_block[0]["category_type"],
            "order": sorted_block[0]["order"],
            "anno_id": sorted_block[0]["anno_id"],
            "text": text,
            "merge_list": sorted_block,
        }
        saved_element_dict[sorted_block[0]["category_type"]].append(merged_block)

    return dict(saved_element_dict)


# ---------------------------------------------------------------------------
# Content accessor for GT items
# ---------------------------------------------------------------------------


def _get_gt_raw_content(item: Dict) -> str:
    """Return the raw text/latex/html content of a GT layout_det item."""
    cat = item.get("category_type", "")
    if cat in _FORMULA_CATEGORIES:
        return item.get("latex") or item.get("text") or ""
    if cat in _TABLE_CATEGORIES:
        return item.get("html") or item.get("latex") or ""
    return item.get("text") or ""


# ---------------------------------------------------------------------------
# Prediction parsing  (ported from MDPBench utils/extract.py::md_tex_filter)
# ---------------------------------------------------------------------------

_RE_HTML_TABLE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_RE_LATEX_TABLE = re.compile(
    r"\\begin\{(?:table|tabular)\*?\}[\s\S]*?\\end\{(?:table|tabular)\*?\}",
    re.IGNORECASE,
)
_RE_DISPLAY_MATH_DD = re.compile(r"\$\$([\s\S]*?)\$\$")
_RE_DISPLAY_MATH_BK = re.compile(r"\\\[([\s\S]*?)\\\]")
_RE_MD_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_RE_INLINE_MATH = re.compile(r"\$(?!\$)(.+?)\$(?!\$)")
_RE_PAREN_MATH = re.compile(r"\\\((.+?)\\\)")


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code-fence markers at line boundaries (mirrors remove_markdown_fences).

    Uses re.MULTILINE so that ^ and $ anchor to line boundaries, not just the
    full string — this means only fences that appear at the start of a line are
    removed, matching the original MDPBench data_preprocess.remove_markdown_fences
    behaviour.  The old single-pass regex had no MULTILINE flag and would
    incorrectly strip ``` sequences embedded inside content.
    """
    text = re.sub(r"^```markdown\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```html\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```latex\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```md\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\n?", "", text, flags=re.MULTILINE)
    return text


def _replace_repeated_chars(s: str) -> str:
    """Normalise consecutive underscores and spaces (mirrors replace_repeated_chars).

    ____+ → ____ (exactly 4 underscores)
     {4,} → four spaces
    Applied to the raw model output before any extraction, as in md_tex_filter.
    """
    s = re.sub(r"_{4,}", "____", s)
    s = re.sub(r" {4,}", "    ", s)
    return s


def _md_table_to_html(md_block: str) -> str:
    lines = [line.strip() for line in md_block.strip().splitlines() if line.strip()]
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in lines if not re.match(r"^[\s|:-]+$", line)]
    if not rows:
        return ""
    html = "<table>"
    for i, row in enumerate(rows):
        tag = "th" if i == 0 else "td"
        html += "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in row) + "</tr>"
    html += "</table>"
    return html


def _parse_prediction_simplified(content: str) -> Dict[str, List[Dict]]:
    """Parse model Markdown output into typed pred-item lists.

    Each item is a dict compatible with the original md_tex_filter output:
        {"category_type": str, "content": str, "position": [start, end]}

    Keys in returned dict:
        "text_all"          — plain text paragraphs
        "equation_isolated" — display-math LaTeX; inline math has fine_category_type='equation_inline'
        "html_table"        — HTML <table> strings
        "latex_table"       — LaTeX tabular/table strings
        "md2html_table"     — Markdown pipe-tables converted to HTML
    """
    content = _strip_code_fences(content.strip())
    # Normalise repeated underscores / spaces (mirrors replace_repeated_chars in md_tex_filter)
    content = _replace_repeated_chars(content)
    # Strip wrapping html/body tags (mirrors md_tex_filter line 119)
    content = content.replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "")
    result: Dict[str, List[Dict]] = defaultdict(list)

    remaining = content

    def _record(cat, text, start, end, fine_category_type=None):
        item: Dict = {
            "category_type": cat,
            "content": text,
            "position": [start, end],
        }
        if fine_category_type:
            item["fine_category_type"] = fine_category_type
        result[cat].append(item)

    # 1. HTML tables
    for m in _RE_HTML_TABLE.finditer(remaining):
        _record("html_table", m.group(0), m.start(), m.end())
    remaining = _RE_HTML_TABLE.sub("\x00", remaining)

    # 2. LaTeX tables
    for m in _RE_LATEX_TABLE.finditer(remaining):
        _record("latex_table", m.group(0), m.start(), m.end())
    remaining = _RE_LATEX_TABLE.sub("\x00", remaining)

    # 3. Markdown pipe-tables → HTML
    md_blocks: List[str] = []
    current: List[str] = []
    for line in remaining.splitlines():
        if _RE_MD_TABLE_ROW.match(line):
            current.append(line)
        else:
            if current:
                md_blocks.append("\n".join(current))
                current = []
    if current:
        md_blocks.append("\n".join(current))
    for block in md_blocks:
        html = _md_table_to_html(block)
        if html:
            start = remaining.find(block)
            _record("md2html_table", html, start, start + len(block))
        remaining = remaining.replace(block, "\x00", 1)

    # 4. Display math  $$...$$
    for m in _RE_DISPLAY_MATH_DD.finditer(remaining):
        latex = m.group(1).strip()
        if latex:
            _record("equation_isolated", latex, m.start(), m.end())
    remaining = _RE_DISPLAY_MATH_DD.sub("\x00", remaining)

    # 5. Display math  \[...\]
    for m in _RE_DISPLAY_MATH_BK.finditer(remaining):
        latex = m.group(1).strip()
        if latex:
            _record("equation_isolated", latex, m.start(), m.end())
    remaining = _RE_DISPLAY_MATH_BK.sub("\x00", remaining)

    # 5.5. Inline math  $...$  and  \(...\)
    # Mirrors md_tex_filter: extracted as equation_isolated with fine_category_type='equation_inline'
    # but NOT removed from remaining — they also appear in text paragraphs (step 6).
    for m in _RE_INLINE_MATH.finditer(remaining):
        latex = m.group(1).strip()
        if latex:
            _record("equation_isolated", f"\\[{latex}\\]", m.start(), m.end(), fine_category_type="equation_inline")
    for m in _RE_PAREN_MATH.finditer(remaining):
        latex = m.group(1).strip()
        if latex:
            _record("equation_isolated", f"\\[{latex}\\]", m.start(), m.end(), fine_category_type="equation_inline")

    # 6. Remaining text → paragraphs
    # Mirrors md_tex_filter split logic exactly:
    #   primary  : split by double-newline (≥2 \n) or \x00 table marker
    #   fallback : if no double-newlines exist, split by single \n
    #              (md_tex_filter: "some models do not use double newlines")
    # Inline math $...$ stays in text content as-is; textblock2unicode
    # normalises it during matching.
    _remaining_text = remaining.replace("\x00", "")
    _split_pat = r"\n{2,}|\x00" if "\n\n" in _remaining_text else r"\n|\x00"
    pos = 0
    for para in re.split(_split_pat, remaining):
        plain = para.strip()
        if plain:
            start = remaining.find(para, pos)
            _record("text_all", plain, start, start + len(para))
            pos = start + len(para)

    return dict(result)


def _parse_prediction(content: str) -> Dict[str, List[Dict]]:
    """Parse predictions with the reference MDPBench ``md_tex_filter``.

    The implementation is vendored under this task so lmms-eval does not need
    a separate MDPBench checkout at runtime.
    """
    from lmms_eval.tasks.mdpbench.extract import md_tex_filter

    return dict(md_tex_filter(content))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_RE_NON_WORD = re.compile(r"[^\w一-鿿]")

_LATEX_STYLE_PATS = [
    (re.compile(r"\\math(?:bf|rm|it|cal|bb)\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\boldsymbol\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\text(?:bf|it)?\{([^}]*)\}"), r"\1"),
]
_LATEX_REMOVE_PATS = [
    re.compile(p)
    for p in [
        r"\\(?:left|right)",
        r"\\(?:quad|qquad|,|:|;|!)",
        r"\\displaystyle",
        r"\\tag\{[^}]*\}",
        r"\\[hv]space\{[^}]*\}",
        r"\\begin\{[^}]*\}",
        r"\\end\{[^}]*\}",
    ]
]


def _normalize_text(s: str) -> str:
    """clean_string equivalent: strip escape sequences and non-word/non-CJK chars."""
    s = s.replace("\\t", "").replace("\\n", "").replace("\t", "").replace("\n", "").replace("/t", "").replace("/n", "")
    return _RE_NON_WORD.sub("", s)


def _normalize_formula(s: str) -> str:
    """normalized_formula equivalent: strip delimiters and LaTeX decorators."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"^\$\$|\$\$$", "", s.strip())
    s = re.sub(r"^\\\[|\\\]$", "", s.strip())
    for pat, repl in _LATEX_STYLE_PATS:
        s = pat.sub(repl, s)
    for pat in _LATEX_REMOVE_PATS:
        s = pat.sub(" ", s)
    s = s.lower()
    return " ".join(s.split())


def _normalize_html_table(html: str) -> str:
    """Faithful replica of MDPBench utils/data_preprocess.py::normalized_html_table.

    Steps (matching the original exactly):
      1. th→td, unwrap thead, replace <math alttext> with $alttext$, unwrap span
      2. html.unescape() + NFKC normalisation
      3. Extract inner table content via regex
      4. Strip style/height/width/align/class attributes
      5. Remove <tbody> tags
      6. Collapse whitespace
      7. Wrap in <html><body><table border="1" >…</table></body></html>
      8. clean_table: remove sup/sub/span/div/p/<spandata…>/colgroup tags
    """
    import html as _html_mod

    # Quick guard: if no <table tag, return as-is
    if "<table" not in html.replace(" ", "").replace("'", '"'):
        return html

    # --- step 1: BeautifulSoup DOM transforms ---
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for th in soup.find_all("th"):
            th.name = "td"
        for thead in soup.find_all("thead"):
            thead.unwrap()
        for math_tag in soup.find_all("math"):
            alttext = math_tag.get("alttext", "")
            math_tag.replace_with(f"${alttext}$")
        for span in soup.find_all("span"):
            span.unwrap()
        processed = str(soup)
    except Exception:
        processed = html

    # --- step 2: unescape HTML entities + NFKC ---
    table_res = _html_mod.unescape(processed).replace("\n", "")
    table_res = unicodedata.normalize("NFKC", table_res).strip()

    # --- step 3: extract inner table content ---
    tables = re.findall(r"<table\b[^>]*>(.*)</table>", table_res, re.DOTALL | re.IGNORECASE)
    if not tables:
        return html
    table_res = "".join(tables)

    # --- step 4: strip presentational attributes ---
    table_res = re.sub(r' style=".*?"', "", table_res)
    table_res = re.sub(r' height=".*?"', "", table_res)
    table_res = re.sub(r' width=".*?"', "", table_res)
    table_res = re.sub(r' align=".*?"', "", table_res)
    table_res = re.sub(r' class=".*?"', "", table_res)

    # --- step 5: remove tbody ---
    table_res = re.sub(r"</?tbody>", "", table_res)

    # --- step 6: collapse whitespace ---
    table_res = re.sub(r"\s+", " ", table_res)

    # --- step 7: wrap ---
    table_res = f'<html><body><table border="1" >{table_res}</table></body></html>'

    # --- step 8: clean_table — remove inline formatting tags ---
    table_res = table_res.replace("<sup>", "").replace("</sup>", "")
    table_res = table_res.replace("<sub>", "").replace("</sub>", "")
    table_res = table_res.replace("<span>", "").replace("</span>", "")
    table_res = table_res.replace("<div>", "").replace("</div>", "")
    table_res = table_res.replace("<p>", "").replace("</p>", "")
    table_res = table_res.replace('<spandata-span-identity="">', "")
    table_res = re.sub(r"<colgroup>.*?</colgroup>", "", table_res)

    return table_res


# ---------------------------------------------------------------------------
# Matching  (simplified match_gt2pred_simple — Hungarian on mixed pool)
# ---------------------------------------------------------------------------


def _match_tables_raw(
    gt_items: List[Dict],
    pred_items: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Match tables using raw HTML/LaTeX content — mirrors match_gt2pred_simple.

    Original match_gt2pred_simple for line_type='html_table' sets
    norm_gt_lines = gt_lines and norm_pred_lines = pred_lines (no normalisation),
    so edit distance is computed on raw HTML strings.  This function replicates
    that behaviour exactly.
    """
    if not gt_items:
        return [], list(pred_items)

    # Raw content: GT uses item['html'] (or 'content' fallback), pred uses item['content']
    gt_raw = [str(_get_gt_raw_content(g)) for g in gt_items]
    pred_raw = [str(p.get("content", "")) for p in pred_items] if pred_items else []

    if not pred_items:
        results = []
        for i, g in enumerate(gt_items):
            results.append(
                {
                    "gt_idx": [i],
                    "gt": gt_raw[i],
                    "norm_gt": gt_raw[i],
                    "pred_idx": [""],
                    "pred": "",
                    "norm_pred": "",
                    "gt_category_type": g.get("category_type", ""),
                    "pred_category_type": "",
                    "gt_position": [g.get("order", 0)],
                    "pred_position": "",
                    "edit": 1.0,
                }
            )
        return results, []

    # Cost matrix on raw strings (no normalisation)
    cost: List[List[float]] = []
    for gr in gt_raw:
        row = []
        for pr in pred_raw:
            ml = max(len(gr), len(pr))
            row.append(0.0 if ml == 0 else Levenshtein.distance(gr, pr) / ml)
        cost.append(row)

    pairs = _run_hungarian(cost)
    matched_gt = {p[0] for p in pairs}
    matched_pred = {p[1] for p in pairs}

    results: List[Dict] = []
    for gi, pi, c in pairs:
        g, p = gt_items[gi], pred_items[pi]
        results.append(
            {
                "gt_idx": [gi],
                "gt": gt_raw[gi],
                "norm_gt": gt_raw[gi],
                "pred_idx": [pi],
                "pred": pred_raw[pi],
                "norm_pred": pred_raw[pi],
                "gt_category_type": g.get("category_type", ""),
                "pred_category_type": p.get("category_type", ""),
                "gt_position": [g.get("order", 0)],
                "pred_position": p.get("position", [0])[0],
                "edit": c,
            }
        )

    for i, g in enumerate(gt_items):
        if i not in matched_gt:
            results.append(
                {
                    "gt_idx": [i],
                    "gt": gt_raw[i],
                    "norm_gt": gt_raw[i],
                    "pred_idx": [""],
                    "pred": "",
                    "norm_pred": "",
                    "gt_category_type": g.get("category_type", ""),
                    "pred_category_type": "",
                    "gt_position": [g.get("order", 0)],
                    "pred_position": "",
                    "edit": 1.0,
                }
            )

    unmatched_pred = [p for i, p in enumerate(pred_items) if i not in matched_pred]
    return results, unmatched_pred


def _normalize_item(item: Dict) -> str:
    """Choose normaliser based on category_type."""
    cat = item.get("category_type", "")
    content = item.get("content", "") or _get_gt_raw_content(item)
    if cat in ("equation_isolated", "equation_inline"):
        return _normalize_formula(content)
    return _normalize_text(content)


def _edit_distance_matrix(
    gt_items: List[Dict],
    pred_items: List[Dict],
) -> List[List[float]]:
    norm_gt = [_normalize_item(g) for g in gt_items]
    norm_pred = [_normalize_item(p) for p in pred_items]
    cost = []
    for ng in norm_gt:
        row = []
        for np_ in norm_pred:
            ml = max(len(ng), len(np_))
            row.append(0.0 if ml == 0 else Levenshtein.distance(ng, np_) / ml)
        cost.append(row)
    return cost


def _run_hungarian(
    cost: List[List[float]],
) -> List[Tuple[int, int, float]]:
    """Return (gt_idx, pred_idx, cost) pairs.  Falls back to greedy.

    No cost threshold is applied here — mirrors match_gt2pred_simple which
    accepts every Hungarian assignment regardless of edit distance.  The 0.7
    threshold that exists in match_gt2pred_quick applies only to the text
    matching path (handled inside match_algo.py), not to table matching.
    """
    if not cost or not cost[0]:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        arr = np.array(cost)
        ri, ci = linear_sum_assignment(arr)
        return [(int(r), int(c), float(arr[r, c])) for r, c in zip(ri, ci)]
    except ImportError:
        eval_logger.warning("scipy unavailable, using greedy match for MDPBench")
        return _greedy_match(cost)


def _greedy_match(cost: List[List[float]]) -> List[Tuple[int, int, float]]:
    """Greedy fallback (scipy unavailable). No threshold — matches all pairs."""
    pairs = sorted(
        [(cost[i][j], i, j) for i in range(len(cost)) for j in range(len(cost[0]))],
    )
    used_gt, used_pred = set(), set()
    out = []
    for c, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        out.append((i, j, c))
        used_gt.add(i)
        used_pred.add(j)
    return out


def _match_elements(
    gt_items: List[Dict],
    pred_items: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Match GT items against pred items via Hungarian algorithm.

    Returns:
        matched      — list of match dicts (includes unmatched GT with edit=1)
        unmatched_pred — pred items with no GT match
    """
    if not gt_items:
        return [], list(pred_items)

    norm_gt = [_normalize_item(g) for g in gt_items]
    norm_pred = [_normalize_item(p) for p in pred_items] if pred_items else []

    if not pred_items:
        pairs = []
    else:
        cost = _edit_distance_matrix(gt_items, pred_items)
        pairs = _run_hungarian(cost)

    matched_gt = {p[0] for p in pairs}
    matched_pred = {p[1] for p in pairs}

    results: List[Dict] = []

    # Matched pairs
    for gi, pi, c in pairs:
        g, p = gt_items[gi], pred_items[pi]
        results.append(
            {
                "gt_idx": [gi],
                "gt": _get_gt_raw_content(g),
                "norm_gt": norm_gt[gi],
                "pred_idx": [pi],
                "pred": p.get("content", ""),
                "norm_pred": norm_pred[pi],
                "gt_category_type": g.get("category_type", ""),
                "pred_category_type": p.get("category_type", ""),
                "gt_position": [g.get("order", 0)],
                "pred_position": p.get("position", [0])[0],
                "edit": c,
            }
        )

    # Unmatched GT → full penalty
    for i, g in enumerate(gt_items):
        if i not in matched_gt:
            ng = norm_gt[i]
            results.append(
                {
                    "gt_idx": [i],
                    "gt": _get_gt_raw_content(g),
                    "norm_gt": ng,
                    "pred_idx": [""],
                    "pred": "",
                    "norm_pred": "",
                    "gt_category_type": g.get("category_type", ""),
                    "pred_category_type": "",
                    "gt_position": [g.get("order", 0)],
                    "pred_position": "",
                    "edit": 1.0,
                }
            )

    unmatched_pred = [p for i, p in enumerate(pred_items) if i not in matched_pred]
    return results, unmatched_pred


# ---------------------------------------------------------------------------
# Formula reclassification
# Faithful replica of MDPBench get_matched_elements L212-230
# ---------------------------------------------------------------------------


def _reclassify_formulas(
    formula_matches: List[Dict],
    text_matches: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Move formula matches whose prediction is non-formula into text pool.

    When the model outputs plain text instead of LaTeX for a formula element,
    the GT LaTeX is converted to Unicode text (via pylatexenc) and the pair
    is re-scored as a text edit-distance comparison instead of CDM.

    Mirrors MDPBench End2EndDataset.get_matched_elements() L212-230.
    """
    formula_clean: List[Dict] = []
    formula_as_text: List[Dict] = []

    for item in formula_matches:
        pred_cat = item.get("pred_category_type", "")
        if pred_cat not in ("equation_inline", "equation_isolated", ""):
            # Prediction was plain text — convert GT LaTeX → text
            gt_latex = item.get("gt", "")
            try:
                from pylatexenc.latex2text import LatexNodes2Text

                if not hasattr(_reclassify_formulas, "_l2t"):
                    _reclassify_formulas._l2t = LatexNodes2Text()
                item["gt"] = _reclassify_formulas._l2t.latex_to_text(gt_latex)
            except Exception as exc:
                eval_logger.warning(f"latex2text failed for formula reclassification: {exc}")
            from lmms_eval.tasks.mdpbench.data_preprocess import clean_string

            item["norm_gt"] = clean_string(item["gt"])
            formula_as_text.append(item)
        else:
            formula_clean.append(item)

    # The reference performs this after accumulating every page. Its complete
    # benchmark text pool is non-empty, so each reclassified formula contributes
    # to text scoring even when its own page has no other text match.
    if formula_as_text:
        text_matches.extend(formula_as_text)
    return formula_clean, text_matches


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _compute_text_edit_from_matches(matches: List[Dict]) -> Dict[str, float]:
    """Weighted normalised edit distance for text matches.

    Uses pre-computed norm_gt / norm_pred from the match dicts.
    Weight = max(len(norm_gt), len(norm_pred)) per pair.
    Unmatched GT items (pred='') contribute weight=len(norm_gt), edit=1.

    Returns {"distance": float[0,1], "weight": float}
    """
    total_dist = 0.0
    total_weight = 0.0
    for item in matches:
        ng = item.get("norm_gt", "")
        np_ = item.get("norm_pred", "")
        ml = max(len(ng), len(np_)) if np_ else len(ng)
        if ml == 0:
            continue
        dist = Levenshtein.distance(ng, np_) if np_ else len(ng)
        total_dist += dist
        total_weight += ml
    if total_weight == 0:
        return {"distance": 0.0, "weight": 0.0}
    return {"distance": total_dist / total_weight, "weight": total_weight}


def _compute_formula_cdm_from_matches(
    matches: List[Dict],
    img_id: str,
) -> Dict[str, object]:
    """CDM F1 score for formula matches.  Unmatched GT formulas score 0.

    Returns {"score": float[0,1], "count": int}
    """
    if not matches:
        return {"score": 0.0, "count": 0, "items": []}

    evaluator = _get_cdm_evaluator()
    total_f1 = 0.0
    count = 0
    scored_items = []

    for i, item in enumerate(matches):
        gt_latex = item.get("gt", "")
        gt_latex = gt_latex.lstrip("$$").rstrip("$$").strip().lstrip("$").rstrip("$").strip()
        pred_latex = item.get("pred", "")
        pred_latex = pred_latex.split("```latex")[-1].split("```")[0]
        pred_latex = pred_latex.lstrip("$$").rstrip("$$").strip().lstrip("$").rstrip("$").strip()

        item_score = 0.0
        if not gt_latex or not pred_latex:
            count += 1
            scored_items.append({"gt_idx": item.get("gt_idx", [""]), "score": item_score})
            continue
        try:
            r = evaluator.evaluate(gt_latex=gt_latex, pred_latex=pred_latex, img_id=f"{img_id}_f{i}")
            item_score = float(r.get("F1_score", 0.0))
            total_f1 += item_score
            count += 1
        except Exception as exc:
            eval_logger.warning(f"CDM failed for {img_id}: {exc}")
            count += 1
        scored_items.append({"gt_idx": item.get("gt_idx", [""]), "score": item_score})

    if count == 0:
        return {"score": 0.0, "count": 0, "items": []}
    return {"score": total_f1 / count, "count": count, "items": scored_items}


def _compute_teds(gt_html: str, pred_html: str) -> float:
    """TEDS for one table pair using the original MDPBench implementation."""
    if not gt_html or not pred_html:
        return 0.0
    try:
        from lmms_eval.tasks.mdpbench.data_preprocess import normalized_table

        scorer = _get_teds_scorer()
        # RecognitionEnd2EndTableDataset.normalize_data() normalizes matched
        # tables immediately before call_TEDS in the reference pipeline.
        norm_pred = normalized_table(pred_html, "html")
        norm_gt = normalized_table(gt_html)
        return scorer.evaluate(norm_pred, norm_gt)
    except Exception:
        # Original call_TEDS assigns zero on any parsing/scoring failure.
        return 0.0


def _compute_table_teds_from_matches(matches: List[Dict]) -> Dict[str, float]:
    """Average TEDS over table match pairs.  Unmatched GT tables score 0.

    Returns {"teds": float[0,1], "count": int}
    """
    if not matches:
        return {"teds": 0.0, "count": 0}
    total_teds = 0.0
    count = 0
    for item in matches:
        gt_html = item.get("gt", "")
        pred_html = item.get("pred", "")
        total_teds += _compute_teds(gt_html, pred_html) if pred_html else 0.0
        count += 1
    if count == 0:
        return {"teds": 0.0, "count": 0}
    return {"teds": total_teds / count, "count": count}


# ---------------------------------------------------------------------------
# process_results  (called once per sample by lmms-eval)
# ---------------------------------------------------------------------------


def mdpbench_process_results(doc, results):
    """Score one document page.  Orchestrates the full MDPBench pipeline.

    Step 1 : Extract GT elements via _get_page_elements (faithful replica).
    Step 2 : Parse model Markdown output.
    Step 3 : Match tables separately; choose HTML or LaTeX format by count.
    Step 4 : Match GT text+formula pool against pred text+formula pool jointly.
    Step 5 : Split match results by GT category type.
    Step 6 : Reclassify formula matches whose prediction was plain text.
    Step 7 : Filter ignore categories from text scoring pool.
    Step 8 : Compute text edit distance, formula CDM, table TEDS.
    Step 9 : Compute per-page overall score and build payload for aggregation.
    """
    prediction = results[0] if results else ""
    lang = _detect_language(doc)
    digital = _is_digital(doc)
    img_id = _get_image_id(doc)
    answer = _load_answer(doc)

    # --- Step 1: GT elements (faithful replica) ---
    gt_page_elements = _get_page_elements(answer)

    # --- Step 2: Parse prediction ---
    pred_dataset = _parse_prediction(prediction)

    # --- Step 3: Table matching with per-page format selection ---
    # Mirrors process_get_matched_elements L294-309 exactly.
    # Per-page vote: latex_table_len vs effective_html (html + md2html, because
    # the original stores md2html tables under the 'html_table' key).
    # The original's "global" format decision (L243-248) is dead code that always
    # selects the html list (which already contains latex items with pred=""),
    # so the effective behaviour is purely per-page.
    gt_tables_raw = sorted(
        gt_page_elements.get("table", []),
        key=lambda x: x.get("order", 0),
    )
    latex_table_len = len(pred_dataset.get("latex_table", []))
    html_table_len = len(pred_dataset.get("html_table", []))
    md_html_len = len(pred_dataset.get("md2html_table", []))
    effective_html = html_table_len + md_html_len

    unmatched_table_pred: List[Dict] = []
    table_matches: List[Dict] = []

    if gt_tables_raw:
        from lmms_eval.tasks.mdpbench.match import match_gt2pred_simple

        if latex_table_len == 0 and effective_html == 0:
            # No table predictions at all → all GT unmatched
            table_matches, unmatched_table_pred = match_gt2pred_simple(gt_tables_raw, [], "html_table", img_id)
            table_matches = [x for x in table_matches if x["gt_idx"] != [""]]
        elif latex_table_len > effective_html:
            pred_tables = pred_dataset.get("latex_table", [])
            table_matches_raw, unmatched_table_pred = match_gt2pred_simple(gt_tables_raw, pred_tables, "latex_table", img_id)
            # The reference later moves all LaTeX matches into the HTML pool
            # with every prediction field blanked and edit distance set to 1.
            for m in table_matches_raw:
                for key in list(m):
                    if "pred" in key:
                        m[key] = ""
                m["edit"] = 1
            table_matches = [x for x in table_matches_raw if x["gt_idx"] != [""]]
        else:
            pred_tables = pred_dataset.get("html_table", []) + pred_dataset.get("md2html_table", [])
            table_matches_raw, unmatched_table_pred = match_gt2pred_simple(gt_tables_raw, pred_tables, "html_table", img_id)
            table_matches = [x for x in table_matches_raw if x["gt_idx"] != [""]]

    # --- Step 4: Mixed text+formula matching ---
    # Build GT mix: all _GT_MIX_CATEGORIES sorted by reading order
    gt_mix: List[Dict] = []
    for cat in _GT_MIX_CATEGORIES:
        gt_mix.extend(gt_page_elements.get(cat, []))
    gt_mix.sort(key=lambda x: x.get("order") or 0)

    # Build pred mix: everything except table types, plus unmatched table preds.
    # match_gt2pred_simple already decomposes unmatched HTML tables into
    # text_all cell items, exactly as the reference implementation does.
    pred_mix: List[Dict] = []
    for cat, items in pred_dataset.items():
        if cat not in ("html_table", "latex_table", "md2html_table"):
            pred_mix.extend(items)

    if unmatched_table_pred:
        pred_mix.extend(unmatched_table_pred)

    from func_timeout import FunctionTimedOut, func_timeout

    from lmms_eval.tasks.mdpbench.match import match_gt2pred_simple
    from lmms_eval.tasks.mdpbench.match_quick import match_gt2pred_quick

    try:
        all_matches = func_timeout(30, match_gt2pred_quick, args=(gt_mix, pred_mix, "text_all", img_id))
    except FunctionTimedOut:
        all_matches, _ = match_gt2pred_simple(gt_mix, pred_mix, "text_all", img_id)

    # --- Step 5: Split by GT category ---
    plain_text_match: List[Dict] = []
    formula_match: List[Dict] = []

    for item in all_matches:
        cat = item.get("gt_category_type", "")
        if cat in _FORMULA_CATEGORIES:
            formula_match.append(item)
        elif cat:  # any named text category (including ignore ones)
            plain_text_match.append(item)

    # Remove extra predictions (gt_idx == [""]) from formula matches
    # (mirrors L332: display_formula_match_s = [x for x if x['gt_idx'] != [""]])
    formula_match = [x for x in formula_match if x["gt_idx"] != [""]]

    # --- Step 6: Formula reclassification ---
    formula_match, plain_text_match = _reclassify_formulas(formula_match, plain_text_match)

    # --- Step 7: Filter ignore categories from text scoring ---
    # (mirrors filtered_out_ignore L340)
    text_score_matches = [x for x in plain_text_match if x.get("gt_category_type", "") not in _IGNORE_SCORE_CATEGORIES]

    # --- Step 8: Compute per-element scores ---
    text_result = _compute_text_edit_from_matches(text_score_matches)
    formula_result = _compute_formula_cdm_from_matches(formula_match, img_id)
    table_result = _compute_table_teds_from_matches(table_matches)

    text_ed = text_result["distance"]
    formula_cdm = formula_result["score"]
    table_teds = table_result["teds"]

    # --- Step 9: Per-page overall ---
    has_text = any(x.get("gt_category_type", "") not in _IGNORE_SCORE_CATEGORIES for x in plain_text_match) or bool(gt_page_elements.get("text_block"))
    # After reclassification, only formulas that remain in the CDM pool count
    # as a present formula task. The official result file contains no formula
    # entry when all GT formulas were reclassified as text.
    has_formula = bool(formula_match)
    has_table = bool(table_matches) or bool(gt_tables_raw)

    components = []
    if has_text:
        components.append(max(0.0, min(100.0, (1.0 - text_ed) * 100)))
    if has_formula:
        components.append(max(0.0, min(100.0, formula_cdm * 100)))
    if has_table:
        components.append(max(0.0, min(100.0, table_teds * 100)))
    overall = sum(components) / len(components) if components else 0.0

    payload = {
        "text_ed": text_ed,
        "text_weight": text_result["weight"],
        "has_text": has_text,
        "formula_cdm": formula_cdm,
        "formula_count": formula_result["count"],
        "formula_items": formula_result["items"],
        "has_formula": has_formula,
        "table_teds": table_teds,
        "table_count": table_result["count"],
        "has_table": has_table,
        "overall": overall,
        "lang": lang,
        "is_digital": digital,
    }

    metric_names = (
        "mdpbench_overall",
        "mdpbench_text_edit",
        "mdpbench_formula_cdm",
        "mdpbench_table_teds",
        "mdpbench_overall_digital",
        "mdpbench_overall_photo",
        "mdpbench_overall_latin",
        "mdpbench_overall_nonlatin",
        "mdpbench_overall_de",
        "mdpbench_overall_en",
        "mdpbench_overall_es",
        "mdpbench_overall_fr",
        "mdpbench_overall_id",
        "mdpbench_overall_it",
        "mdpbench_overall_nl",
        "mdpbench_overall_pt",
        "mdpbench_overall_vi",
        "mdpbench_overall_ar",
        "mdpbench_overall_hi",
        "mdpbench_overall_jp",
        "mdpbench_overall_ko",
        "mdpbench_overall_ru",
        "mdpbench_overall_th",
        "mdpbench_overall_zh",
        "mdpbench_overall_zh_t",
    )
    return {name: payload for name in metric_names}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _agg_text_edit(
    results,
    *,
    lang_filter=None,
    digital_filter=None,
) -> float:
    # Mirrors original Edit_dist: up_total_avg.mean() = simple mean of per-page
    # scores (each page score = sum(edits) / sum(max_lens) on that page).
    page_scores = []
    for r in results:
        if not r["has_text"]:
            continue
        if lang_filter and r["lang"] not in lang_filter:
            continue
        if digital_filter is not None and r["is_digital"] != digital_filter:
            continue
        page_scores.append(r["text_ed"])
    return sum(page_scores) / len(page_scores) if page_scores else 0.0


def _agg_formula_cdm(
    results,
    *,
    lang_filter=None,
    digital_filter=None,
) -> float:
    total_score = total_count = 0
    for r in results:
        if not r["has_formula"]:
            continue
        if lang_filter and r["lang"] not in lang_filter:
            continue
        if digital_filter is not None and r["is_digital"] != digital_filter:
            continue
        total_score += r["formula_cdm"] * r["formula_count"]
        total_count += r["formula_count"]
    return total_score / total_count if total_count else 0.0


def _agg_table_teds(
    results,
    *,
    lang_filter=None,
    digital_filter=None,
) -> float:
    total_teds = total_count = 0
    for r in results:
        if not r["has_table"]:
            continue
        if lang_filter and r["lang"] not in lang_filter:
            continue
        if digital_filter is not None and r["is_digital"] != digital_filter:
            continue
        total_teds += r["table_teds"] * r["table_count"]
        total_count += r["table_count"]
    return total_teds / total_count if total_count else 0.0


def _agg_overall(
    results,
    *,
    lang_filter=None,
    digital_filter=None,
) -> float:
    """Aggregate overall score following the original MDPBench scoring scheme.

    Mirrors ``compute_present_tasks_overall`` in MDPBench tools/calculate_scores.py:
      1. Per-page overall = mean of whichever task scores are present on that page
         (already stored in r["overall"] by process_results).
      2. Final score = simple mean of per-page overall scores.

    This differs from averaging global-level task metrics: it weights each *page*
    equally rather than each *task type* equally, so pages without formulas/tables
    are not penalised by the global formula/table scores.
    """
    filtered = [r for r in results if (lang_filter is None or r["lang"] in lang_filter) and (digital_filter is None or r["is_digital"] == digital_filter)]
    if not filtered:
        return 0.0

    page_scores = []
    for r in filtered:
        actual_page_components = []
        if r["has_text"]:
            actual_page_components.append(max(0.0, min(100.0, (1.0 - r["text_ed"]) * 100)))
        if r["has_table"]:
            actual_page_components.append(max(0.0, min(100.0, r["table_teds"] * 100)))

        # tools/calculate_scores.py strips only singleton keys like "_[3]".
        # A merged gt_idx such as [9, 16] therefore becomes a separate
        # formula-only pseudo-page in the official aggregate. Reproduce that
        # observable behavior exactly.
        singleton_formula_scores = []
        merged_formula_buckets = defaultdict(list)
        for item in r.get("formula_items", []):
            gt_idx = item.get("gt_idx", [""])
            score = max(0.0, min(100.0, float(item.get("score", 0.0)) * 100))
            if isinstance(gt_idx, list) and len(gt_idx) == 1:
                singleton_formula_scores.append(score)
            else:
                merged_formula_buckets[str(gt_idx)].append(score)

        if singleton_formula_scores:
            actual_page_components.append(sum(singleton_formula_scores) / len(singleton_formula_scores))
        if actual_page_components:
            page_scores.append(sum(actual_page_components) / len(actual_page_components))
        for scores in merged_formula_buckets.values():
            page_scores.append(sum(scores) / len(scores))

    if not page_scores:
        return 0.0
    return sum(page_scores) / len(page_scores)


def _agg_language_group_average(results, languages) -> float:
    """Reproduce MDPBench's reported Latin/Non-Latin group average.

    ``tools/calculate_scores.py`` first rounds every language score to one
    decimal place, then takes an unweighted mean across the languages in the
    group and rounds that mean to one decimal place as well.
    """
    language_scores = []
    for language in sorted(languages):
        if not any(r.get("lang") == language for r in results):
            continue
        language_scores.append(round(_agg_overall(results, lang_filter={language}), 1))
    if not language_scores:
        return 0.0
    return round(sum(language_scores) / len(language_scores), 1)


# ---------------------------------------------------------------------------
# Public aggregation entry points (referenced from mdpbench.yaml)
# ---------------------------------------------------------------------------

# ── Overall ──────────────────────────────────────────────────────────────────


def mdpbench_aggregate_overall(results, args=None):
    return _agg_overall(results)


def mdpbench_aggregate_overall_digital(results, args=None):
    return _agg_overall(results, digital_filter=True)


def mdpbench_aggregate_overall_photo(results, args=None):
    return _agg_overall(results, digital_filter=False)


# ── Latin group + per language ────────────────────────────────────────────────


def mdpbench_aggregate_overall_latin(results, args=None):
    return _agg_language_group_average(results, _LATIN_LANGS)


def mdpbench_aggregate_overall_de(results, args=None):
    return _agg_overall(results, lang_filter={"DE"})


def mdpbench_aggregate_overall_en(results, args=None):
    return _agg_overall(results, lang_filter={"EN"})


def mdpbench_aggregate_overall_es(results, args=None):
    return _agg_overall(results, lang_filter={"ES"})


def mdpbench_aggregate_overall_fr(results, args=None):
    return _agg_overall(results, lang_filter={"FR"})


def mdpbench_aggregate_overall_id(results, args=None):
    return _agg_overall(results, lang_filter={"ID"})


def mdpbench_aggregate_overall_it(results, args=None):
    return _agg_overall(results, lang_filter={"IT"})


def mdpbench_aggregate_overall_nl(results, args=None):
    return _agg_overall(results, lang_filter={"NL"})


def mdpbench_aggregate_overall_pt(results, args=None):
    return _agg_overall(results, lang_filter={"PT"})


def mdpbench_aggregate_overall_vi(results, args=None):
    return _agg_overall(results, lang_filter={"VI"})


# ── Non-Latin group + per language ────────────────────────────────────────────


def mdpbench_aggregate_overall_nonlatin(results, args=None):
    return _agg_language_group_average(results, _NON_LATIN_LANGS)


def mdpbench_aggregate_overall_ar(results, args=None):
    return _agg_overall(results, lang_filter={"AR"})


def mdpbench_aggregate_overall_hi(results, args=None):
    return _agg_overall(results, lang_filter={"HI"})


def mdpbench_aggregate_overall_jp(results, args=None):
    return _agg_overall(results, lang_filter={"JP"})


def mdpbench_aggregate_overall_ko(results, args=None):
    return _agg_overall(results, lang_filter={"KO"})


def mdpbench_aggregate_overall_ru(results, args=None):
    return _agg_overall(results, lang_filter={"RU"})


def mdpbench_aggregate_overall_th(results, args=None):
    return _agg_overall(results, lang_filter={"TH"})


def mdpbench_aggregate_overall_zh(results, args=None):
    return _agg_overall(results, lang_filter={"ZH"})


def mdpbench_aggregate_overall_zh_t(results, args=None):
    return _agg_overall(results, lang_filter={"ZH-T"})


# ── Element-type scores (kept for internal diagnostics) ───────────────────────


def mdpbench_aggregate_text_edit(results, args=None):
    return _agg_text_edit(results)


def mdpbench_aggregate_formula_cdm(results, args=None):
    return _agg_formula_cdm(results)


def mdpbench_aggregate_table_teds(results, args=None):
    return _agg_table_teds(results)

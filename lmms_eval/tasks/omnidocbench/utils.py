"""OmniDocBench evaluation utilities.

Implements the official OmniDocBench document-to-markdown evaluation pipeline:
1. Prompt model to convert document image to structured markdown.
2. Parse predicted markdown into typed elements (text, formula, table).
3. Extract ground truth elements from dataset annotations.
4. Match predicted elements to GT via Hungarian algorithm on edit distance.
5. Score: text edit distance, formula edit distance, table TEDS.
6. Overall = average of (1-text_ed)*100, table_teds*100, (1-formula_ed)*100.

Reference: https://github.com/opendatalab/OmniDocBench
"""

import io
import json
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import Levenshtein
from loguru import logger as eval_logger
from PIL import Image

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_DOC_TO_MARKDOWN_PROMPT = (
    "You are an AI assistant specialized in converting PDF images to Markdown format. "
    "Output ONLY the converted markdown — no explanations, no thinking process, no commentary. "
    "Rules: "
    "1. Recognize all text accurately and convert to Markdown. "
    "2. Convert mathematical formulas to LaTeX (inline: $...$, display: $$...$$). "
    "3. Convert tables to HTML format. "
    "4. Ignore figures and images. "
    "5. Maintain the original document structure and reading order."
)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _to_rgb(image_obj: Any) -> Optional[Image.Image]:
    import base64

    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    if isinstance(image_obj, str) and len(image_obj) > 100:
        # Base64-encoded image string (common in TSV datasets)
        try:
            raw = base64.b64decode(image_obj)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass
    return None


def omnidocbench_doc_to_visual(doc):
    visuals = []
    for key in ["image", "page_image", "document_image"]:
        if key in doc:
            img = _to_rgb(doc[key])
            if img is not None:
                visuals.append(img)
    for key in ["images", "page_images", "document_images", "pages"]:
        value = doc.get(key)
        if isinstance(value, list):
            for item in value:
                img = _to_rgb(item)
                if img is not None:
                    visuals.append(img)
    return visuals


def omnidocbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _DOC_TO_MARKDOWN_PROMPT


def omnidocbench_doc_to_target(doc):
    """Return raw answer JSON string as target (not used for scoring)."""
    return doc.get("answer", "")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _detect_language(doc) -> str:
    """Detect 'en', 'cn', or 'mixed' from the answer JSON."""
    answer_raw = doc.get("answer", "")
    if isinstance(answer_raw, list):
        answer_raw = answer_raw[0] if answer_raw else ""
    if not isinstance(answer_raw, str):
        return "mixed"
    try:
        answer = json.loads(answer_raw)
    except (json.JSONDecodeError, TypeError):
        return "mixed"

    page_info = answer.get("page_info") or {}
    page_attr = page_info.get("page_attribute") or {}
    page_lang = page_attr.get("language", "")
    if "chinese" in page_lang:
        return "cn"
    if "english" in page_lang:
        return "en"

    # Fallback: element-level majority vote
    counts = Counter()
    for det in answer.get("layout_dets", []):
        attr = det.get("attribute") or {}
        lang_val = attr.get("text_language", "") or attr.get("language", "")
        if "chinese" in lang_val:
            counts["cn"] += 1
        elif "english" in lang_val:
            counts["en"] += 1
    if not counts:
        return "mixed"
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

# Category types that map to each element kind
_TEXT_CATEGORIES = {"text_block", "title", "text_caption", "header", "footer", "page_number", "footnote"}
_FORMULA_CATEGORIES = {"equation_isolated", "equation_inline"}
_TABLE_CATEGORIES = {"table"}


def _extract_gt_elements(doc) -> Dict[str, List[str]]:
    """Parse ground truth into {'text': [...], 'formula': [...], 'table': [...]}."""
    answer_raw = doc.get("answer", "")
    if isinstance(answer_raw, list):
        answer_raw = answer_raw[0] if answer_raw else ""
    if not isinstance(answer_raw, str):
        return {"text": [], "formula": [], "table": []}

    try:
        answer = json.loads(answer_raw)
    except (json.JSONDecodeError, TypeError):
        return {"text": [], "formula": [], "table": []}

    layout_dets = answer.get("layout_dets", [])
    relations = []
    extra = answer.get("extra") or {}
    for rel in extra.get("relation", []):
        if rel.get("relation_type") == "truncated":
            relations.append((rel["source_anno_id"], rel["target_anno_id"]))

    # Build merge groups: union-find for truncated relations
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for src, tgt in relations:
        union(src, tgt)

    # Group elements by anno_id merge group
    groups = {}
    for det in layout_dets:
        aid = det.get("anno_id", id(det))
        root = find(aid)
        if root not in groups:
            groups[root] = []
        groups[root].append(det)

    # Sort within each group by order, then collect
    texts = []
    formulas = []
    tables = []

    for root, dets in groups.items():
        dets.sort(key=lambda d: d.get("order") or 0)
        cat = dets[0].get("category_type", "")

        if cat in _TEXT_CATEGORIES:
            merged = " ".join(d.get("text", "") for d in dets if d.get("text"))
            if merged.strip():
                texts.append((dets[0].get("order") or 0, merged.strip()))
        elif cat in _FORMULA_CATEGORIES:
            merged = " ".join(d.get("latex", "") for d in dets if d.get("latex"))
            if merged.strip():
                formulas.append((dets[0].get("order") or 0, merged.strip()))
        elif cat in _TABLE_CATEGORIES:
            merged = " ".join(d.get("html", "") for d in dets if d.get("html"))
            if merged.strip():
                tables.append((dets[0].get("order") or 0, merged.strip()))

    # Sort by reading order
    texts.sort(key=lambda x: x[0])
    formulas.sort(key=lambda x: x[0])
    tables.sort(key=lambda x: x[0])

    return {
        "text": [t[1] for t in texts],
        "formula": [f[1] for f in formulas],
        "table": [t[1] for t in tables],
    }


# ---------------------------------------------------------------------------
# Prediction markdown parsing
# ---------------------------------------------------------------------------

# Regex for display math: $$...$$ or \[...\]
_DISPLAY_MATH_RE = re.compile(
    r"\$\$(.+?)\$\$"  # $$...$$
    r"|\\\[(.+?)\\\]",  # \[...\]
    re.DOTALL,
)

# Regex for HTML tables
_HTML_TABLE_RE = re.compile(r"<table[\s>].*?</table>", re.DOTALL | re.IGNORECASE)

# Regex for markdown pipe tables (header + separator + data rows)
_MD_TABLE_RE = re.compile(
    r"((?:^\|.+\|[ \t]*\n)"  # header row
    r"(?:^\|[\s:|-]+\|[ \t]*\n)"  # separator row
    r"(?:^\|.+\|[ \t]*(?:\n|$))+)",  # data rows
    re.MULTILINE,
)


def _md_table_to_html(md_table: str) -> str:
    """Convert a markdown pipe table to simple HTML."""
    lines = md_table.strip().split("\n")
    if len(lines) < 2:
        return ""

    def parse_row(line):
        cells = line.strip().strip("|").split("|")
        return [c.strip() for c in cells]

    header = parse_row(lines[0])
    # Skip separator line (lines[1])
    data_rows = [parse_row(line) for line in lines[2:] if line.strip()]

    html = "<table><thead><tr>"
    for cell in header:
        html += f"<th>{cell}</th>"
    html += "</tr></thead><tbody>"
    for row in data_rows:
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


def _strip_reasoning_prefix(text: str) -> str:
    """Strip plain-text reasoning preambles from model output.

    Reasoning models may emit "Thinking Process:" or similar before the
    actual markdown.  This removes everything up to and including the
    first markdown-content indicator (heading, table, formula, or a line
    that looks like document text rather than meta-commentary).
    """
    # Quick check: if output starts with actual content, skip stripping
    stripped = text.lstrip()
    if not stripped:
        return text
    first = stripped[0]
    if first in ("#", "|", "<", "$", "\\") or (first.isalnum() and "Thinking" not in stripped[:20]):
        return text

    # Look for the end of reasoning block: a blank line followed by content
    # that looks like markdown (heading, table, list, or CJK text)
    content_start = re.search(
        r"\n\s*\n"  # blank line separator
        r"(?="  # followed by content-like patterns:
        r"[#|<$\\]"  # heading, table, HTML, math
        r"|[\u4e00-\u9fff]"  # CJK characters
        r"|(?:[A-Z][a-z])"  # Capitalized word (likely document text)
        r")",
        text,
    )
    if content_start:
        return text[content_start.end() :].lstrip("\n")

    # Fallback: if "Here is" or "```" marker found, take everything after
    for marker in ["Here is the", "Here's the", "```markdown", "```"]:
        idx = text.find(marker)
        if idx != -1:
            after = text[idx + len(marker) :].lstrip("\n")
            return after

    return text


def _parse_prediction(text: str) -> Dict[str, List[str]]:
    """Parse model markdown output into typed element lists."""
    if not text:
        return {"text": [], "formula": [], "table": []}

    # Strip any reasoning preamble before parsing
    text = _strip_reasoning_prefix(text)

    tables = []
    formulas = []

    # Strip code fences
    text = re.sub(r"```(?:markdown|html|latex)?\s*\n?", "", text)
    text = text.replace("```", "")

    remaining = text

    # 1. Extract HTML tables
    for m in _HTML_TABLE_RE.finditer(remaining):
        tables.append(m.group(0))
    remaining = _HTML_TABLE_RE.sub("\n", remaining)

    # 2. Extract markdown pipe tables -> convert to HTML
    for m in _MD_TABLE_RE.finditer(remaining):
        html = _md_table_to_html(m.group(0))
        if html:
            tables.append(html)
    remaining = _MD_TABLE_RE.sub("\n", remaining)

    # 3. Extract display math
    for m in _DISPLAY_MATH_RE.finditer(remaining):
        content = m.group(1) or m.group(2)
        if content and content.strip():
            formulas.append(content.strip())
    remaining = _DISPLAY_MATH_RE.sub("\n", remaining)

    # 4. Remaining text: split into non-empty paragraphs
    text_blocks = []
    for para in re.split(r"\n\s*\n|\n", remaining):
        # Strip inline math markers but keep content
        para = re.sub(r"\$(.+?)\$", r"\1", para)
        para = para.strip()
        # Skip headings markers but keep text
        para = re.sub(r"^#{1,6}\s*", "", para)
        if para and len(para) > 1:
            text_blocks.append(para)

    return {"text": text_blocks, "formula": formulas, "table": tables}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return any(start <= cp <= end for start, end in _CJK_RANGES)


def _normalize_text(s: str) -> str:
    """Normalize text for edit distance: lowercase, strip non-alnum (keep CJK), collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    # Keep alphanumeric and CJK characters
    out = []
    for ch in s:
        if ch.isalnum() or _is_cjk(ch):
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


# LaTeX styling commands to strip
_LATEX_STRIP_CMDS = [
    r"\\text\{([^}]*)\}",
    r"\\mathrm\{([^}]*)\}",
    r"\\mathbf\{([^}]*)\}",
    r"\\mathit\{([^}]*)\}",
    r"\\mathsf\{([^}]*)\}",
    r"\\mathcal\{([^}]*)\}",
    r"\\mathbb\{([^}]*)\}",
    r"\\boldsymbol\{([^}]*)\}",
    r"\\textbf\{([^}]*)\}",
    r"\\textit\{([^}]*)\}",
]

# Commands to remove entirely (with their arguments)
_LATEX_REMOVE_CMDS = [
    r"\\hspace\{[^}]*\}",
    r"\\vspace\{[^}]*\}",
    r"\\tag\{[^}]*\}",
    r"\\label\{[^}]*\}",
    r"\\quad",
    r"\\qquad",
    r"\\,",
    r"\\;",
    r"\\!",
    r"\\ ",
]


def _normalize_formula(s: str) -> str:
    """Normalize LaTeX formula for edit distance comparison."""
    s = unicodedata.normalize("NFKC", s)
    # Strip begin/end environment wrappers
    s = re.sub(r"\\begin\{[^}]*\}", "", s)
    s = re.sub(r"\\end\{[^}]*\}", "", s)
    # Strip styling commands, keep content
    for pat in _LATEX_STRIP_CMDS:
        s = re.sub(pat, r"\1", s)
    # Remove spacing/tag commands
    for pat in _LATEX_REMOVE_CMDS:
        s = re.sub(pat, " ", s)
    s = s.lower()
    # Collapse whitespace
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Matching via Hungarian algorithm
# ---------------------------------------------------------------------------

_MATCH_COST_THRESHOLD = 0.7  # Pairs with cost above this are treated as unmatched


def _edit_distance_matrix(gt_list: List[str], pred_list: List[str], normalize_fn) -> List[List[float]]:
    """Compute normalized edit distance matrix between GT and predicted elements."""
    n_gt = len(gt_list)
    n_pred = len(pred_list)
    norm_gt = [normalize_fn(s) for s in gt_list]
    norm_pred = [normalize_fn(s) for s in pred_list]

    cost = []
    for i in range(n_gt):
        row = []
        for j in range(n_pred):
            g, p = norm_gt[i], norm_pred[j]
            max_len = max(len(g), len(p))
            if max_len == 0:
                row.append(0.0)
            else:
                dist = Levenshtein.distance(g, p)
                row.append(dist / max_len)
        cost.append(row)
    return cost


def _hungarian_match(gt_list: List[str], pred_list: List[str], normalize_fn) -> List[Tuple[int, int, float]]:
    """Match GT to predicted elements using Hungarian algorithm.

    Returns list of (gt_idx, pred_idx, normalized_edit_distance) for valid matches.
    """
    if not gt_list or not pred_list:
        return []

    cost = _edit_distance_matrix(gt_list, pred_list, normalize_fn)

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost_array = np.array(cost)
        gt_indices, pred_indices = linear_sum_assignment(cost_array)
        matches = []
        for gi, pi in zip(gt_indices, pred_indices):
            c = cost_array[gi, pi]
            if c <= _MATCH_COST_THRESHOLD:
                matches.append((int(gi), int(pi), float(c)))
        return matches
    except ImportError:
        eval_logger.warning("scipy not available, falling back to greedy matching")
        return _greedy_match(cost)


def _greedy_match(cost: List[List[float]]) -> List[Tuple[int, int, float]]:
    """Greedy fallback when scipy is not available."""
    n_gt = len(cost)
    n_pred = len(cost[0]) if cost else 0
    used_pred = set()
    matches = []

    # Sort all (gt, pred) pairs by cost
    pairs = []
    for i in range(n_gt):
        for j in range(n_pred):
            pairs.append((cost[i][j], i, j))
    pairs.sort()

    used_gt = set()
    for c, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        if c > _MATCH_COST_THRESHOLD:
            break
        matches.append((i, j, c))
        used_gt.add(i)
        used_pred.add(j)

    return matches


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def _compute_text_edit_distance(gt_texts: List[str], pred_texts: List[str]) -> Dict[str, float]:
    """Compute weighted edit distance for text elements on a single page.

    Returns dict with 'distance' (weighted NED) and 'weight' (total max_len).
    """
    if not gt_texts:
        return {"distance": 0.0, "weight": 0.0}

    matches = _hungarian_match(gt_texts, pred_texts, _normalize_text)
    matched_gt = {m[0] for m in matches}

    total_dist = 0.0
    total_max_len = 0.0

    # Matched pairs
    for gi, pi, _ in matches:
        g = _normalize_text(gt_texts[gi])
        p = _normalize_text(pred_texts[pi])
        dist = Levenshtein.distance(g, p)
        max_len = max(len(g), len(p))
        total_dist += dist
        total_max_len += max_len

    # Unmatched GT elements get full penalty
    for i, gt in enumerate(gt_texts):
        if i not in matched_gt:
            g = _normalize_text(gt)
            total_dist += len(g)
            total_max_len += len(g)

    if total_max_len == 0:
        return {"distance": 0.0, "weight": 0.0}

    return {"distance": total_dist / total_max_len, "weight": total_max_len}


def _compute_formula_edit_distance(gt_formulas: List[str], pred_formulas: List[str]) -> Dict[str, float]:
    """Compute weighted edit distance for formula elements on a single page."""
    if not gt_formulas:
        return {"distance": 0.0, "weight": 0.0}

    matches = _hungarian_match(gt_formulas, pred_formulas, _normalize_formula)
    matched_gt = {m[0] for m in matches}

    total_dist = 0.0
    total_max_len = 0.0

    for gi, pi, _ in matches:
        g = _normalize_formula(gt_formulas[gi])
        p = _normalize_formula(pred_formulas[pi])
        dist = Levenshtein.distance(g, p)
        max_len = max(len(g), len(p))
        total_dist += dist
        total_max_len += max_len

    for i, gt in enumerate(gt_formulas):
        if i not in matched_gt:
            g = _normalize_formula(gt)
            total_dist += len(g)
            total_max_len += len(g)

    if total_max_len == 0:
        return {"distance": 0.0, "weight": 0.0}

    return {"distance": total_dist / total_max_len, "weight": total_max_len}


def _compute_teds(gt_html: str, pred_html: str) -> float:
    """Compute Tree Edit Distance Similarity for a single table pair.

    Uses the TEDS implementation from ocrbench_v2 if available,
    otherwise falls back to normalized string edit distance.
    """
    if not gt_html or not pred_html:
        return 0.0

    try:
        from lmms_eval.tasks.ocrbench_v2.TEDS_metric import TEDS, wrap_html_table

        teds_scorer = TEDS(structure_only=False, n_jobs=1)
        gt_wrapped = wrap_html_table(gt_html)
        pred_wrapped = wrap_html_table(pred_html)
        score = teds_scorer.evaluate(pred_wrapped, gt_wrapped)
        return max(0.0, min(1.0, score))
    except Exception:
        # Fallback: normalized string edit distance on HTML
        gt_norm = re.sub(r"\s+", " ", gt_html.strip().lower())
        pred_norm = re.sub(r"\s+", " ", pred_html.strip().lower())
        max_len = max(len(gt_norm), len(pred_norm))
        if max_len == 0:
            return 1.0
        dist = Levenshtein.distance(gt_norm, pred_norm)
        return 1.0 - dist / max_len


def _compute_table_teds(gt_tables: List[str], pred_tables: List[str]) -> Dict[str, float]:
    """Compute average TEDS for matched table pairs on a single page."""
    if not gt_tables:
        return {"teds": 0.0, "count": 0}

    # For table matching, use normalized HTML string edit distance
    def _normalize_html(s):
        return re.sub(r"\s+", " ", s.strip().lower())

    matches = _hungarian_match(gt_tables, pred_tables, _normalize_html)
    matched_gt = {m[0] for m in matches}

    total_teds = 0.0
    count = 0

    for gi, pi, _ in matches:
        score = _compute_teds(gt_tables[gi], pred_tables[pi])
        total_teds += score
        count += 1

    # Unmatched GT tables get score 0
    for i in range(len(gt_tables)):
        if i not in matched_gt:
            count += 1
            # total_teds += 0.0

    if count == 0:
        return {"teds": 0.0, "count": 0}

    return {"teds": total_teds / count, "count": count}


# ---------------------------------------------------------------------------
# Process results (called per-sample by lmms-eval)
# ---------------------------------------------------------------------------


def omnidocbench_process_results(doc, results):
    """Score a single document page."""
    prediction = results[0] if results else ""
    lang = _detect_language(doc)

    gt = _extract_gt_elements(doc)
    pred = _parse_prediction(prediction)

    text_result = _compute_text_edit_distance(gt["text"], pred["text"])
    formula_result = _compute_formula_edit_distance(gt["formula"], pred["formula"])
    table_result = _compute_table_teds(gt["table"], pred["table"])

    # Per-page metrics
    text_ed = text_result["distance"]
    formula_ed = formula_result["distance"]
    table_teds = table_result["teds"]

    # Compute overall for this page:
    # Average of available component scores
    components = []
    if gt["text"]:
        components.append((1.0 - text_ed) * 100)
    if gt["table"]:
        components.append(table_teds * 100)
    if gt["formula"]:
        components.append((1.0 - formula_ed) * 100)
    overall = sum(components) / len(components) if components else 0.0

    # Build payload for aggregation
    payload = {
        "text_ed": text_ed,
        "text_weight": text_result["weight"],
        "has_text": len(gt["text"]) > 0,
        "formula_ed": formula_ed,
        "formula_weight": formula_result["weight"],
        "has_formula": len(gt["formula"]) > 0,
        "table_teds": table_teds,
        "table_count": table_result["count"],
        "has_table": len(gt["table"]) > 0,
        "overall": overall,
        "lang": lang,
    }

    return {
        "omnidocbench_overall": payload,
        "omnidocbench_text_edit": payload,
        "omnidocbench_table_teds": payload,
        "omnidocbench_formula_edit": payload,
        "omnidocbench_text_edit_en": payload,
        "omnidocbench_text_edit_cn": payload,
        "omnidocbench_table_teds_en": payload,
        "omnidocbench_table_teds_cn": payload,
        "omnidocbench_formula_edit_en": payload,
        "omnidocbench_formula_edit_cn": payload,
    }


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


def _aggregate_text_edit(results, lang_filter=None):
    """Weighted average of text edit distances across pages."""
    total_dist = 0.0
    total_weight = 0.0
    for r in results:
        if not r["has_text"]:
            continue
        if lang_filter and r["lang"] not in lang_filter:
            continue
        total_dist += r["text_ed"] * r["text_weight"]
        total_weight += r["text_weight"]
    if total_weight == 0:
        return 0.0
    return total_dist / total_weight


def _aggregate_formula_edit(results, lang_filter=None):
    """Weighted average of formula edit distances across pages."""
    total_dist = 0.0
    total_weight = 0.0
    for r in results:
        if not r["has_formula"]:
            continue
        if lang_filter and r["lang"] not in lang_filter:
            continue
        total_dist += r["formula_ed"] * r["formula_weight"]
        total_weight += r["formula_weight"]
    if total_weight == 0:
        return 0.0
    return total_dist / total_weight


def _aggregate_table_teds(results, lang_filter=None):
    """Weighted average of table TEDS across pages."""
    total_teds = 0.0
    total_count = 0
    for r in results:
        if not r["has_table"]:
            continue
        if lang_filter and r["lang"] not in lang_filter:
            continue
        total_teds += r["table_teds"] * r["table_count"]
        total_count += r["table_count"]
    if total_count == 0:
        return 0.0
    return total_teds / total_count


def _aggregate_overall(results, lang_filter=None):
    """Overall score: average of component scores across pages."""
    filtered = [r for r in results if (not lang_filter or r["lang"] in lang_filter)]
    if not filtered:
        return 0.0

    text_ed = _aggregate_text_edit(filtered)
    formula_ed = _aggregate_formula_edit(filtered)
    table_teds_val = _aggregate_table_teds(filtered)

    has_text = any(r["has_text"] for r in filtered)
    has_formula = any(r["has_formula"] for r in filtered)
    has_table = any(r["has_table"] for r in filtered)

    components = []
    if has_text:
        components.append((1.0 - text_ed) * 100)
    if has_table:
        components.append(table_teds_val * 100)
    if has_formula:
        components.append((1.0 - formula_ed) * 100)

    return sum(components) / len(components) if components else 0.0


# --- Public aggregation entry points (called by lmms-eval YAML) ---


def omnidocbench_aggregate_overall(results, args=None):
    return _aggregate_overall(results)


def omnidocbench_aggregate_text_edit(results, args=None):
    return _aggregate_text_edit(results)


def omnidocbench_aggregate_table_teds(results, args=None):
    return _aggregate_table_teds(results)


def omnidocbench_aggregate_formula_edit(results, args=None):
    return _aggregate_formula_edit(results)


def omnidocbench_aggregate_text_edit_en(results, args=None):
    return _aggregate_text_edit(results, lang_filter={"en"})


def omnidocbench_aggregate_text_edit_cn(results, args=None):
    return _aggregate_text_edit(results, lang_filter={"cn"})


def omnidocbench_aggregate_table_teds_en(results, args=None):
    return _aggregate_table_teds(results, lang_filter={"en"})


def omnidocbench_aggregate_table_teds_cn(results, args=None):
    return _aggregate_table_teds(results, lang_filter={"cn"})


def omnidocbench_aggregate_formula_edit_en(results, args=None):
    return _aggregate_formula_edit(results, lang_filter={"en"})


def omnidocbench_aggregate_formula_edit_cn(results, args=None):
    return _aggregate_formula_edit(results, lang_filter={"cn"})

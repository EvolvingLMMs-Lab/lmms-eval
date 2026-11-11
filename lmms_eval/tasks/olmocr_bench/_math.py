# Adapted from allenai/olmocr at f7cfe4c22098b154c76b6ec950d1c0a464eecf8d.
# Copyright Allen Institute for Artificial Intelligence.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Pinned KaTeX rendering and geometry comparison with lazy, in-memory caching.

DOM extraction and comparison match the official evaluator. Browser state belongs
only to a dedicated worker thread; importing this module never starts a browser
or writes a cache to disk.
"""

import json
import os
import re
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache

from lmms_eval.imports import require_package

require_package("playwright", extras="olmocr_bench", feature="olmOCR-Bench equation rendering")
require_package("bs4", extras="olmocr_bench", feature="olmOCR-Bench MathML comparison")
require_package("lxml", extras="olmocr_bench", feature="olmOCR-Bench MathML comparison")
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

_worker = threading.local()


def _close_browser(playwright, browser):
    try:
        browser.close()
    finally:
        playwright.stop()


class _BrowserOwner:
    """Release Playwright in its owning thread when the worker exits."""

    def __init__(self):
        playwright = sync_playwright().start()
        try:
            self.browser = playwright.chromium.launch()
        except PlaywrightError as error:
            playwright.stop()
            message = "olmOCR-Bench requires Chromium. Install it with: "
            message += "uv run --extra olmocr_bench python -m playwright install chromium"
            raise RuntimeError(message) from error
        weakref.finalize(self, _close_browser, playwright, self.browser)


@lru_cache(maxsize=1)
def _executor():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="olmocr-equations")


def _render_in_worker(equation, bg_color, text_color, font_size, use_cache, debug_dom):
    if not hasattr(_worker, "owner"):
        _worker.owner = _BrowserOwner()
        _worker.cache = OrderedDict()
    key = (equation, bg_color, text_color, font_size)
    if use_cache and key in _worker.cache:
        _worker.cache.move_to_end(key)
        return _worker.cache[key]
    context = _worker.owner.browser.new_context(viewport={"width": 800, "height": 400})
    try:
        rendered = _do_render(context, equation, bg_color, text_color, font_size, debug_dom)
    finally:
        context.close()
    if use_cache and rendered and not rendered.error:
        _worker.cache[key] = rendered
        if len(_worker.cache) > 4096:
            _worker.cache.popitem(last=False)
    return rendered


def render_equation(equation, bg_color="white", text_color="black", font_size=24, use_cache=True, debug_dom=False):
    return (
        _executor()
        .submit(
            _render_in_worker,
            equation,
            bg_color,
            text_color,
            font_size,
            use_cache,
            debug_dom,
        )
        .result()
    )


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class SpanInfo:
    text: str
    bounding_box: BoundingBox


@dataclass(frozen=True)
class RenderedEquation:
    mathml: str
    spans: list[SpanInfo]
    error: str | None = None  # New field to store error messages if rendering fails


def _do_render(context, equation, bg_color, text_color, font_size, debug_dom):
    """
    Internal rendering function that uses a provided browser context.
    """
    # Escape the equation for use in a JavaScript string.
    escaped_equation = json.dumps(equation)

    # Get local paths for KaTeX files.
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_katex")
    katex_css_path = os.path.join(script_dir, "katex.min.css")
    katex_js_path = os.path.join(script_dir, "katex.min.js")

    if not os.path.exists(katex_css_path) or not os.path.exists(katex_js_path):
        raise FileNotFoundError(
            f"KaTeX files not found. Please ensure katex.min.css and katex.min.js are in {script_dir}",
        )

    # Create a new page.
    page = context.new_page()

    # Basic HTML structure for rendering.
    page_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: {bg_color};
                color: {text_color};
            }}
            #equation-container {{
                padding: 0;
                font-size: {font_size}px;
            }}
        </style>
    </head>
    <body>
        <div id="equation-container"></div>
    </body>
    </html>
    """
    page.set_content(page_html)
    page.add_style_tag(path=katex_css_path)
    page.add_script_tag(path=katex_js_path)
    page.wait_for_load_state("networkidle", timeout=0)

    katex_loaded = page.evaluate("typeof katex !== 'undefined'")
    if not katex_loaded:
        page.close()
        raise RuntimeError("KaTeX library failed to load. Check your katex.min.js file.")

    try:
        error_message = page.evaluate(f"""
        () => {{
            try {{
                katex.render({escaped_equation}, document.getElementById("equation-container"), {{
                    displayMode: true,
                    throwOnError: true
                }});
                return null;
            }} catch (error) {{
                console.error("KaTeX error:", error.message);
                return error.message;
            }}
        }}
        """)
    except PlaywrightError as ex:
        print(escaped_equation)
        error_message = str(ex)
        page.close()
        raise

    if error_message:
        print(f"Error rendering equation: '{equation}'")
        print(error_message)
        # Return error result
        page.close()
        return RenderedEquation(mathml=error_message, spans=[], error=error_message)

    page.wait_for_selector(".katex", state="attached", timeout=0)

    if debug_dom:
        katex_dom_html = page.evaluate("""
        () => {
            return document.getElementById("equation-container").innerHTML;
        }
        """)
        print("\n===== KaTeX DOM HTML =====")
        print(katex_dom_html)

    # Extract inner-most spans with non-whitespace text.
    spans_info = page.evaluate("""
    () => {
        const spans = Array.from(document.querySelectorAll('span'));
        const list = [];
        spans.forEach(span => {
            if (span.children.length === 0 && /\\S/.test(span.textContent)) {
                const rect = span.getBoundingClientRect();
                list.push({
                    text: span.textContent.trim(),
                    boundingBox: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    }
                });
            }
        });
        return list;
    }
    """)

    if debug_dom:
        print("\n===== Extracted Span Information =====")
        print(spans_info)

    # Extract MathML output (if available) from the KaTeX output.
    mathml = page.evaluate("""
        () => {
            const mathElem = document.querySelector('.katex-mathml math');
            return mathElem ? mathElem.outerHTML : "";
        }
        """)

    page.close()

    rendered_eq = RenderedEquation(
        mathml=mathml,
        spans=[
            SpanInfo(
                text=s["text"],
                bounding_box=BoundingBox(
                    x=s["boundingBox"]["x"],
                    y=s["boundingBox"]["y"],
                    width=s["boundingBox"]["width"],
                    height=s["boundingBox"]["height"],
                ),
            )
            for s in spans_info
        ],
    )

    return rendered_eq


def compare_rendered_equations(reference: RenderedEquation, hypothesis: RenderedEquation) -> bool:
    """
    Compare two RenderedEquation objects.
    First, check if the normalized MathML of the hypothesis is contained within that of the reference.
    If not, perform a neighbor-based matching on the spans.
    """
    from bs4 import BeautifulSoup

    def extract_inner(mathml: str) -> str:
        try:
            soup = BeautifulSoup(mathml, "xml")
            semantics = soup.find("semantics")
            if semantics:
                children = semantics.contents
                inner_parts = [str(child) for child in children if getattr(child, "name", None) != "annotation"]
                return "".join(inner_parts)
            else:
                return str(soup)
        except Exception as e:
            print("Error parsing MathML with BeautifulSoup:", e)
            print(mathml)
            return mathml

    # Filters out all spaces, include unicode zero and partial width spaces
    def normalize(s: str) -> str:
        return re.sub(r"[\s\u200a-\u200d\u2060\ufeff]+", "", s, flags=re.UNICODE)

    reference_inner = normalize(extract_inner(reference.mathml))
    hypothesis_inner = normalize(extract_inner(hypothesis.mathml))
    if reference_inner in hypothesis_inner:
        return True

    H, R = reference.spans, hypothesis.spans

    # Filter out any whitespace only spans and remove whitespace from the spans
    H = [replace(span, text=normalize(span.text)) for span in H if normalize(span.text)]
    R = [replace(span, text=normalize(span.text)) for span in R if normalize(span.text)]

    def expand_span_info(span_info: SpanInfo) -> list[SpanInfo]:
        total_elems = len(span_info.text)
        return [
            SpanInfo(
                c,
                BoundingBox(
                    span_info.bounding_box.x + (span_info.bounding_box.width * index) / total_elems,
                    span_info.bounding_box.y,
                    span_info.bounding_box.width / total_elems,
                    span_info.bounding_box.height,
                ),
            )
            for index, c in enumerate(span_info.text)
        ]

    H = [span for sublist in H for span in expand_span_info(sublist)]
    R = [span for sublist in R for span in expand_span_info(sublist)]

    candidate_map = {}
    for i, hspan in enumerate(H):
        candidate_map[i] = [j for j, rsp in enumerate(R) if rsp.text == hspan.text]
        if not candidate_map[i]:
            return False

    def compute_neighbors(spans, tol=5):
        neighbors = {}
        for i, span in enumerate(spans):
            cx = span.bounding_box.x + span.bounding_box.width / 2
            cy = span.bounding_box.y + span.bounding_box.height / 2
            up = down = left = right = None
            up_dist = down_dist = left_dist = right_dist = None
            for j, other in enumerate(spans):
                if i == j:
                    continue
                ocx = other.bounding_box.x + other.bounding_box.width / 2
                ocy = other.bounding_box.y + other.bounding_box.height / 2
                if ocy < cy and abs(ocx - cx) <= tol:
                    dist = cy - ocy
                    if up is None or dist < up_dist:
                        up = j
                        up_dist = dist
                if ocy > cy and abs(ocx - cx) <= tol:
                    dist = ocy - cy
                    if down is None or dist < down_dist:
                        down = j
                        down_dist = dist
                if ocx < cx and abs(ocy - cy) <= tol:
                    dist = cx - ocx
                    if left is None or dist < left_dist:
                        left = j
                        left_dist = dist
                if ocx > cx and abs(ocy - cy) <= tol:
                    dist = ocx - cx
                    if right is None or dist < right_dist:
                        right = j
                        right_dist = dist
            neighbors[i] = {"up": up, "down": down, "left": left, "right": right}
        return neighbors

    hyp_neighbors = compute_neighbors(H)
    ref_neighbors = compute_neighbors(R)

    n = len(H)
    used = [False] * len(R)
    assignment = {}

    def backtrack(i):
        if i == n:
            return True
        for cand in candidate_map[i]:
            if used[cand]:
                continue
            assignment[i] = cand
            used[cand] = True
            valid = True
            for direction in ["up", "down", "left", "right"]:
                hyp_nb = hyp_neighbors[i].get(direction)
                ref_nb = ref_neighbors[cand].get(direction)
                if hyp_nb is not None:
                    expected_text = H[hyp_nb].text
                    if ref_nb is None:
                        valid = False
                        break
                    if hyp_nb in assignment:
                        if assignment[hyp_nb] != ref_nb:
                            valid = False
                            break
                    else:
                        if R[ref_nb].text != expected_text:
                            valid = False
                            break
            if valid:
                if backtrack(i + 1):
                    return True
            used[cand] = False
            del assignment[i]
        return False

    return backtrack(0)

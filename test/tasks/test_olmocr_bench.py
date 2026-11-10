"""Offline contracts and opt-in Poppler/Chromium integrations for olmOCR-Bench.

Set LMMS_OLMOCR_BENCH_INTEGRATION=1 to exercise the real PDF and KaTeX paths.
Integration prerequisites are required when opted in; missing tools must fail.
"""

import importlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
from PIL import Image

from lmms_eval.api.task import ConfigurableMessagesTask
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.olmocr_bench import utils
from lmms_eval.utils import load_yaml_config

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks" / "olmocr_bench"
YAML_PATH = TASK_DIR / "olmocr_bench.yaml"


@pytest.fixture
def integration() -> None:
    if os.environ.get("LMMS_OLMOCR_BENCH_INTEGRATION") != "1":
        pytest.skip("set LMMS_OLMOCR_BENCH_INTEGRATION=1 for Poppler/Chromium integration")


@pytest.fixture
def table_dependencies() -> None:
    for dependency in ("bs4", "lxml"):
        if os.environ.get("LMMS_OLMOCR_BENCH_INTEGRATION") == "1":
            importlib.import_module(dependency)
        else:
            pytest.importorskip(dependency)


@pytest.fixture
def order_dependencies() -> None:
    if os.environ.get("LMMS_OLMOCR_BENCH_INTEGRATION") == "1":
        importlib.import_module("fuzzysearch")
    else:
        pytest.importorskip("fuzzysearch")


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_yaml_config_and_task_discovery() -> None:
    cfg = load_yaml_config(str(YAML_PATH))
    assert cfg["task"] == "olmocr_bench"
    assert cfg["dataset_path"] == "allenai/olmOCR-bench"
    assert cfg["dataset_kwargs"]["revision"] == "54a96a6fb6a2bd3b297e59869491db4d3625b711"
    assert cfg["dataset_kwargs"]["data_files"] == {"test": "bench_data/arxiv_math.jsonl"}
    assert cfg["test_split"] == "test"
    assert cfg["output_type"] == "generate_until"
    assert cfg["reasoning_tags"] is False
    assert cfg["process_docs"].__name__ == "olmocr_bench_process_docs"
    assert cfg["doc_to_visual"].__name__ == "olmocr_bench_doc_to_visual"
    assert cfg["doc_to_text"].__name__ == "olmocr_bench_doc_to_text"
    assert cfg["process_results"].__name__ == "olmocr_bench_process_results"

    metric_names = [m["metric"] for m in cfg["metric_list"]]
    assert metric_names == list(utils.ALL_METRICS)

    tm = TaskManager("ERROR")
    assert "olmocr_bench" in tm.all_tasks


def test_process_docs_groups_and_interleaves_categories() -> None:
    raw_ds = datasets.Dataset.from_list(
        [
            {
                "id": "m1",
                "pdf": "arxiv_math/p1.pdf",
                "page": 1,
                "type": "math",
                "math": "a + b = c",
            },
            {
                "id": "m2",
                "pdf": "arxiv_math/p1.pdf",
                "page": 1,
                "type": "math",
                "math": "x^2 + y^2 = z^2",
            },
            {
                "id": "t1",
                "pdf": "tables/t1.pdf",
                "page": 1,
                "type": "table",
                "cell": "100",
                "top_heading": "Jan",
            },
            {
                "id": "hf1",
                "pdf": "headers_footers/h1.pdf",
                "page": 1,
                "type": "absent",
                "text": "Page 12",
                "first_n": 50,
            },
            {
                "id": "hf_base",
                "pdf": "headers_footers/h1.pdf",
                "page": 1,
                "type": "baseline",
                "check_disallowed_characters": False,
            },
        ],
    )

    grouped = utils.olmocr_bench_process_docs(raw_ds)
    assert len(grouped) == 3
    assert [row["category"] for row in grouped] == [
        "arxiv_math",
        "table_tests",
        "headers_footers",
    ]

    arxiv_tests = json.loads(grouped[0]["tests_json"])
    assert len(arxiv_tests) == 3
    assert [t["type"] for t in arxiv_tests] == ["math", "math", "baseline"]
    assert [t["group"] for t in arxiv_tests] == ["arxiv_math", "arxiv_math", "baseline"]

    hf_tests = json.loads(grouped[2]["tests_json"])
    # Explicit baseline test in headers_footers.jsonl stays in headers_footers group; no duplicate auto-baseline
    assert len(hf_tests) == 2
    assert [t["type"] for t in hf_tests] == ["absent", "baseline"]
    assert [t["group"] for t in hf_tests] == ["headers_footers", "headers_footers"]


def test_doc_to_visual_text_and_messages_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "sample_page.pdf"
    pdf_path.write_bytes(b"PDF download boundary fixture")
    image_buffer = io.BytesIO()
    Image.new("RGB", (2048, 1024), color=(20, 40, 60)).save(image_buffer, format="PNG")
    commands = []

    def run_poppler(command, **kwargs):
        commands.append(command)
        if command[0] == "pdfinfo":
            return subprocess.CompletedProcess(command, 0, "Page 1 MediaBox: 0 0 200 100\n", "")
        assert command[0] == "pdftoppm"
        return subprocess.CompletedProcess(command, 0, image_buffer.getvalue(), b"")

    monkeypatch.setattr(utils.shutil, "which", lambda executable: f"/bin/{executable}")
    monkeypatch.setattr(utils.subprocess, "run", run_poppler)
    monkeypatch.setattr(utils, "hf_hub_download", lambda **kwargs: str(pdf_path))

    doc = {
        "pdf": "arxiv_math/sample_page.pdf",
        "page": 1,
        "category": "arxiv_math",
        "tests_json": json.dumps([{"id": "b1", "pdf": "p.pdf", "page": 1, "type": "baseline"}]),
    }

    visuals = utils.olmocr_bench_doc_to_visual(doc)
    assert len(visuals) == 1
    assert isinstance(visuals[0], Image.Image)
    assert visuals[0].mode == "RGB"
    assert visuals[0].size == (2048, 1024)
    assert visuals[0].getpixel((0, 0)) == (20, 40, 60)
    assert commands[0][-1] == str(pdf_path)
    assert commands[1][commands[1].index("-r") + 1] == "737.28"
    assert commands[1][commands[1].index("-f") + 1] == "1"
    assert commands[1][commands[1].index("-l") + 1] == "1"

    # Also test direct image dict / bytes fallback
    img_doc = {"image": {"bytes": _png_bytes((10, 20, 30)), "path": "inline.png"}}
    inline_vis = utils.olmocr_bench_doc_to_visual(img_doc)
    assert len(inline_vis) == 1 and inline_vis[0].getpixel((0, 0)) == (10, 20, 30)

    default_prompt = utils.olmocr_bench_doc_to_text(doc)
    assert "Below is the image of one page of a PDF document." in default_prompt
    basic_prompt = utils.olmocr_bench_doc_to_text(
        doc,
        {"prompt_template": "basic", "pre_prompt": "[PRE] ", "post_prompt": " [POST]"},
    )
    assert basic_prompt.startswith("[PRE] Please provide a natural, plain text representation")
    assert basic_prompt.endswith("Convert any tables into Markdown format. [POST]")

    with pytest.raises(ValueError, match="unknown prompt_template"):
        utils.olmocr_bench_doc_to_text(doc, {"prompt_template": "nonexistent"})

    task = ConfigurableMessagesTask.__new__(ConfigurableMessagesTask)
    task._config = type(
        "Cfg",
        (),
        {
            "doc_to_messages": None,
            "doc_to_visual": utils.olmocr_bench_doc_to_visual,
            "doc_to_text": utils.olmocr_bench_doc_to_text,
        },
    )()
    task.doc_to_visual = utils.olmocr_bench_doc_to_visual
    task.doc_to_text = lambda d: utils.olmocr_bench_doc_to_text(d)
    messages = task.doc_to_messages(img_doc)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [item["type"] for item in content] == ["image", "text"]
    assert content[1]["text"] == default_prompt


def test_text_presence_and_order_evaluators(order_dependencies) -> None:
    present_test = {
        "id": "p1",
        "pdf": "doc.pdf",
        "page": 1,
        "type": "present",
        "text": "The quick “brown” fox — jumps",
        "case_sensitive": True,
        "max_diffs": 1,
    }
    assert utils.run_single_pdf_test(
        present_test,
        'Header\nThe **quick** "brown" fox - jumps over the lazy dog.',
    )
    assert not utils.run_single_pdf_test(
        present_test,
        'the QUICK "brown" fox - jumps',
    )
    assert utils.run_single_pdf_test(
        {**present_test, "case_sensitive": False},
        'the QUICK "brown" fox - jumps',
    )

    absent_header = {
        "id": "a1",
        "pdf": "doc.pdf",
        "page": 1,
        "type": "absent",
        "text": "CONFIDENTIAL HEADER",
        "first_n": 30,
        "max_diffs": 0,
    }
    assert utils.run_single_pdf_test(
        absent_header,
        "Clean start of body text that mentions CONFIDENTIAL HEADER much later in the document.",
    )
    assert not utils.run_single_pdf_test(
        absent_header,
        "CONFIDENTIAL HEADER\nClean start of body text.",
    )

    order_test = {
        "id": "o1",
        "pdf": "doc.pdf",
        "page": 1,
        "type": "order",
        "before": "First column paragraph alpha",
        "after": "Second column paragraph beta",
        "max_diffs": 2,
    }
    assert utils.run_single_pdf_test(
        order_test,
        "Intro. First column paragrph alpha comes here, then Second column paragraph beta.",
    )
    assert not utils.run_single_pdf_test(
        order_test,
        "Second column paragraph beta appears before First column paragraph alpha.",
    )


def test_table_parsing_and_table_test_evaluator(table_dependencies) -> None:
    md_table = "| Metric | Q1 | Q2 |\n| --- | --- | --- |\n| Revenue | 120 | 150 |\n| Profit | 40 | 55 |\n"
    md_test = {
        "id": "tb_md",
        "pdf": "t.pdf",
        "page": 1,
        "type": "table",
        "cell": "150",
        "left": "120",
        "down": "55",
        "top_heading": "Q2",
        "left_heading": "Revenue",
    }
    assert utils.run_single_pdf_test(md_test, md_table)
    assert not utils.run_single_pdf_test({**md_test, "top_heading": "Q1"}, md_table)

    html_table = """
    <table>
      <thead>
        <tr>
          <th rowspan="2"></th>
          <th colspan="2">Main Group</th>
        </tr>
        <tr>
          <th>Col A</th>
          <th>Col B</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Row 1</th>
          <td>10<br>units</td>
          <td>20</td>
        </tr>
      </tbody>
    </table>
    """
    html_test = {
        "id": "tb_html",
        "pdf": "t.pdf",
        "page": 1,
        "type": "table",
        "cell": "20",
        "left": "10 units",
        "top_heading": "Main Group",
        "left_heading": "Row 1",
    }
    assert utils.run_single_pdf_test(html_test, html_table)


@pytest.mark.parametrize(
    ("ref_math", "md_content", "expected"),
    [
        ("a+b", r"Inline \(a + b\) equation", True),
        ("c-d", r"Inline \(a + b\) equation", False),
        (
            r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
            r"$$\begin{align*}\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}\end{align*}$$",
            True,
        ),
        (
            r"\left\{ \left( 0_{X},0_{Y},-1\right) \right\}",
            r"\[ \{(0_{X}, 0_{Y}, -1)\} \]",
            True,
        ),
        (
            r"\lambda_g = \lambda_{g'}",
            r"$\lambda_{g}=\lambda_{g^{\prime}}$",
            True,
        ),
        (
            r"T_{ij}^{kl}",
            r"$T^{kl}_{ij}$",
            True,
        ),
        (
            r"C_T \left(u^T_n X^{\text{Test}}_n,\overline{ x}^{\text{Test}}\right)",
            r"$$C_{T}\left(u_{n}^{T} X_{n}^{\text{Test }}, \overline{x}^{\text {Test }}\right)$$",
            True,
        ),
        (
            r"C_{T}\left(u_{n}^{T} X_{n}^{\text {Test }}, \bar{x}^{\text {Test }}\right)",
            r"$$C_T \left(u^T_n X^{\text{Test}}_n,\overline{ \text{x}}^{\text{Test}}\right)$$",
            False,
        ),
        (r"\frac{a}{b}", r"$a / b$", False),
        (r"\frac{\partial f}{\partial x}", r"$\frac{\partial_f}{\partial_x}$", False),
        (r"x_{i,j}", r"$x_{j,i}$", False),
        (r"\phi", r"$\varphi$", False),
        (r"a + b", r"$\color{red}{a} + \color{black}{b}$", True),
        (r"\alpha + \beta", r"$\newcommand{\ab}{\alpha + \beta}\ab$", True),
        (r"\to", r"$\rightarrow$", True),
    ],
)
def test_math_test_evaluator_katex_suite(integration, ref_math: str, md_content: str, expected: bool) -> None:
    test_obj = {"id": "m_case", "pdf": "m.pdf", "page": 1, "type": "math", "math": ref_math}
    assert utils.run_single_pdf_test(test_obj, md_content) is expected


def test_baseline_test() -> None:
    baseline = {"id": "b1", "pdf": "p.pdf", "page": 1, "type": "baseline"}
    assert utils.run_single_pdf_test(baseline, "Valid English paragraph with 123 numbers.")
    assert not utils.run_single_pdf_test(baseline, "   \n... !@# ")
    assert not utils.run_single_pdf_test(baseline, "Valid start " + ("abc" * 35))
    assert not utils.run_single_pdf_test(baseline, "English text with CJK 漢字")
    assert not utils.run_single_pdf_test(baseline, "English text with emoji 😀")
    assert utils.run_single_pdf_test(
        {**baseline, "check_disallowed_characters": False},
        "Allowed CJK 漢字 when check_disallowed_characters is False",
    )

    blank_check = {
        "id": "b_blank",
        "pdf": "p.pdf",
        "page": 1,
        "type": "baseline",
        "max_length": 5,
        "max_length_skips_image_alt_tags": True,
    }
    assert utils.run_single_pdf_test(blank_check, "![diagram description](page_0_0_100_100.png)")
    assert not utils.run_single_pdf_test(blank_check, "Non-blank paragraph text")


@pytest.mark.parametrize("test_type", ["present", "math"])
def test_process_results_json_serializable_and_macro_aggregation(test_type: str, request) -> None:
    if test_type == "math":
        request.getfixturevalue("integration")
    else:
        request.getfixturevalue("table_dependencies")
    doc_math = {
        "pdf": "arxiv_math/m1.pdf",
        "page": 1,
        "category": "arxiv_math",
        "tests_json": json.dumps(
            [
                {
                    "id": "m1",
                    "pdf": "arxiv_math/m1.pdf",
                    "page": 1,
                    "type": test_type,
                    ("math" if test_type == "math" else "text"): "E = mc^2",
                    "group": "arxiv_math",
                },
                {
                    "id": "m2",
                    "pdf": "arxiv_math/m1.pdf",
                    "page": 1,
                    "type": test_type,
                    ("math" if test_type == "math" else "text"): "a^2 + b^2 = c^2",
                    "group": "arxiv_math",
                },
                {
                    "id": "b1",
                    "pdf": "arxiv_math/m1.pdf",
                    "page": 1,
                    "type": "baseline",
                    "group": "baseline",
                },
            ],
        ),
    }
    doc_table = {
        "pdf": "tables/t1.pdf",
        "page": 1,
        "category": "table_tests",
        "tests_json": json.dumps(
            [
                {
                    "id": "t1",
                    "pdf": "tables/t1.pdf",
                    "page": 1,
                    "type": "table",
                    "cell": "42",
                    "top_heading": "Score",
                    "group": "table_tests",
                },
                {
                    "id": "b2",
                    "pdf": "tables/t1.pdf",
                    "page": 1,
                    "type": "baseline",
                    "group": "baseline",
                },
            ],
        ),
    }

    res_math = utils.olmocr_bench_process_results(doc_math, [r"Physics formula: \(E = mc^2\)"])
    res_table = utils.olmocr_bench_process_results(doc_table, ["| Score |\n| --- |\n| 42 |"])

    # Strictly JSON-serializable for 2-process accelerate gather
    json.dumps(res_math)
    json.dumps(res_table)

    assert res_math["olmocr_bench_arxiv_math"] == {"passed": 1, "total": 2}
    assert res_math["olmocr_bench_baseline"] == {"passed": 1, "total": 1}
    assert res_math["olmocr_bench_table_tests"] is None

    assert res_table["olmocr_bench_table_tests"] == {"passed": 1, "total": 1}
    assert res_table["olmocr_bench_baseline"] == {"passed": 1, "total": 1}

    overall_records = [res_math["olmocr_bench_overall"], res_table["olmocr_bench_overall"]]
    assert utils.olmocr_bench_aggregate_arxiv_math(
        [res_math["olmocr_bench_arxiv_math"], res_table["olmocr_bench_arxiv_math"]],
    ) == pytest.approx(0.5)
    assert utils.olmocr_bench_aggregate_table_tests(
        [res_math["olmocr_bench_table_tests"], res_table["olmocr_bench_table_tests"]],
    ) == pytest.approx(1.0)
    assert utils.olmocr_bench_aggregate_baseline(
        [res_math["olmocr_bench_baseline"], res_table["olmocr_bench_baseline"]],
    ) == pytest.approx(1.0)
    assert math.isnan(
        utils.olmocr_bench_aggregate_old_scans(
            [res_math["olmocr_bench_old_scans"], res_table["olmocr_bench_old_scans"]],
        ),
    )
    # Macro-average over active categories: (0.5 + 1.0 + 1.0) / 3 = 5/6
    assert utils.olmocr_bench_aggregate_overall(overall_records) == pytest.approx(5.0 / 6.0)
    assert math.isnan(utils.olmocr_bench_aggregate_overall([]))


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        (r"a\ne b", r"$a\neq b$", True),
        ("a+b", r"$\invalid{a+b}$", False),
        ("a+b", "${a}+{b}$", True),
        ("1", "$10$", True),
        ("x", "$x_1$", True),
    ],
    ids=["command-alias", "invalid-command", "equivalent-braces", "number-extension", "subscript-extension"],
)
def test_math_matches_pinned_browser_oracle(integration, reference: str, prediction: str, expected: bool) -> None:
    """Literal outcomes from official f7cfe4c with KaTeX 0.16.21 and Chromium."""
    test = {"id": "math-oracle", "pdf": "synthetic.pdf", "page": 1, "type": "math", "math": reference}
    assert utils.run_single_pdf_test(test, prediction) is expected


@pytest.mark.parametrize(
    ("fields", "prediction"),
    [
        ({"cell": "A"}, "<table><tr><td>A</td><td>B</td></tr>"),
        (
            {"cell": "OuterInnerTail"},
            "<table><tr><td>Outer<table><tr><td>Inner</td></tr></table>Tail</td><td>B</td></tr></table>",
        ),
        ({"cell": "R", "right": "B"}, '<table><tr><td rowspan="0">R</td><td>A</td></tr><tr><td>B</td></tr></table>'),
    ],
    ids=["unclosed-html", "nested-html", "zero-rowspan"],
)
def test_html_table_matches_pinned_oracle(table_dependencies, fields: dict, prediction: str) -> None:
    """Official f7cfe4c accepts malformed/nested HTML through BeautifulSoup."""
    test = {"id": "html-oracle", "pdf": "synthetic.pdf", "page": 1, "type": "table", **fields}
    assert utils.run_single_pdf_test(test, prediction) is True


@pytest.mark.parametrize(
    "prediction",
    [
        "<think>forbidden</think>\nclean body",
        "unfinished thought</think>forbidden",
        "```markdown\nbody\n```",
        "---\nmetadata: forbidden\n---\nclean body",
        '{"metadata":"forbidden","natural_text":"clean body"}',
        '{"natural_text": "From JSON schema"}',
        "\n".join(
            [
                "<think>CONFIDENTIAL HEADER should be omitted</think>",
                "```markdown",
                "---",
                "primary_language: en",
                "is_rotation_valid: true",
                "rotation_correction: 0",
                "is_table: false",
                "is_diagram: false",
                "---",
                "Clean body paragraph.",
                "```",
            ],
        ),
        "null",
        "  body with whitespace \n",
        "__import__('os').system('false')",
        "1e309",
        "9" * 400,
    ],
)
def test_plain_markdown_prediction_is_not_rewritten(prediction: str) -> None:
    """Official plain response path scores the complete Markdown file."""
    assert utils.clean_model_prediction(prediction) == prediction


def test_none_prediction_becomes_empty_markdown() -> None:
    assert utils.clean_model_prediction(None) == ""


@pytest.mark.parametrize(
    ("test_fields", "prediction", "expected"),
    [
        ({"type": "absent", "text": "forbidden"}, "<think>forbidden</think>\nclean body", 0),
        ({"type": "absent", "text": "forbidden"}, "---\nmetadata: forbidden\n---\nclean body", 0),
        ({"type": "absent", "text": "forbidden"}, '{"metadata":"forbidden","natural_text":"clean body"}', 0),
        ({"type": "present", "text": "body"}, "```markdown\nbody\n```", 1),
        ({"type": "baseline"}, "null", 1),
        ({"type": "baseline"}, "", 0),
        ({"type": "present", "text": "body", "first_n": 1}, " body", 0),
        ({"type": "present", "text": "body", "last_n": 1}, "body ", 0),
    ],
    ids=["think", "yaml", "json", "fence", "literal-null", "empty", "leading-space", "trailing-space"],
)
def test_process_results_scores_raw_markdown(test_fields: dict, prediction: str, expected: int) -> None:
    """Literal expectations from the official f7cfe4c plain response contract."""
    test = {"id": "raw-oracle", "pdf": "synthetic.pdf", "page": 1, "group": "baseline", **test_fields}
    doc = {"pdf": "synthetic.pdf", "page": 1, "category": "baseline", "tests_json": json.dumps([test])}
    result = utils.olmocr_bench_process_results(doc, [prediction])
    assert result["olmocr_bench_baseline"] == {"passed": expected, "total": 1}
    assert result["olmocr_bench_overall"]["passed"] == expected
    json.dumps(result)


@pytest.mark.parametrize("tool", ["pdfinfo", "pdftoppm"])
def test_poppler_command_errors_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
    def run_poppler(command, **kwargs):
        if command[0] == tool:
            stderr = "invalid PDF" if tool == "pdfinfo" else b"invalid PDF"
            return subprocess.CompletedProcess(command, 1, "" if tool == "pdfinfo" else b"", stderr)
        return subprocess.CompletedProcess(command, 0, "Page 1 MediaBox: 0 0 200 100\n", "")

    monkeypatch.setattr(utils.shutil, "which", lambda executable: f"/bin/{executable}")
    monkeypatch.setattr(utils.subprocess, "run", run_poppler)
    with pytest.raises((ValueError, RuntimeError), match="invalid PDF"):
        utils.olmocr_bench_doc_to_visual({"pdf_path": str(tmp_path / f"{tool}.pdf")})


def test_missing_poppler_reports_required_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.shutil, "which", lambda executable: None)
    with pytest.raises(RuntimeError, match="requires Poppler"):
        utils.olmocr_bench_doc_to_visual({"pdf_path": str(tmp_path / "missing-renderer.pdf")})


def test_real_poppler_pdf_to_visual(integration, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample_page.pdf"
    Image.new("RGB", (200, 100), color=(20, 40, 60)).save(pdf_path, format="PDF", resolution=72.0)
    visuals = utils.olmocr_bench_doc_to_visual({"pdf_path": str(pdf_path), "page": 1})
    assert len(visuals) == 1
    assert visuals[0].mode == "RGB"
    assert max(visuals[0].size) == pytest.approx(2048, abs=4)


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize(
    ("fields", "prediction"),
    [
        ({"type": "present", "text": "A", "first_n": 1}, "  A"),
        ({"type": "present", "text": "A", "last_n": 1}, "A  "),
        ({"type": "baseline"}, None),
    ],
    ids=["first-character-space", "last-character-space", "none"],
)
def test_task_process_results_preserves_raw_boundary(fields: dict, prediction: str | None, nested: bool) -> None:
    from lmms_eval.tasks.olmocr_bench.task import OlmocrBenchTask

    task = OlmocrBenchTask.__new__(OlmocrBenchTask)
    task.OUTPUT_TYPE = "generate_until"
    task._config = SimpleNamespace(process_results=utils.olmocr_bench_process_results)
    test = {"id": "boundary", "pdf": "synthetic.pdf", "page": 1, "group": "baseline", **fields}
    doc = {"pdf": "synthetic.pdf", "page": 1, "category": "baseline", "tests_json": json.dumps([test])}
    results = [[prediction]] if nested else [prediction]

    result = task.process_results(doc, results)

    # Official normalization retains one edge space; the requested one-character
    # window contains no A. A missing response also fails the baseline test.
    assert result["olmocr_bench_baseline"] == {"passed": 0, "total": 1}
    assert result["olmocr_bench_overall"]["passed"] == 0
    json.dumps(result)


@pytest.mark.parametrize("protocol", ["simple", "chat"])
def test_task_construct_requests_preserves_backend_protocol(protocol: str) -> None:
    from lmms_eval.tasks.olmocr_bench.task import OlmocrBenchTask

    task = OlmocrBenchTask.__new__(OlmocrBenchTask)
    task.OUTPUT_TYPE = "generate_until"
    task._request_protocol = protocol
    task._config = SimpleNamespace(
        task="olmocr_bench",
        generation_kwargs={"temperature": 0.0, "until": []},
        doc_to_messages=None,
        doc_to_visual=utils.olmocr_bench_doc_to_visual,
        doc_to_text=utils.olmocr_bench_doc_to_text,
    )
    task.lmms_eval_specific_kwargs = None
    metadata = {"task": "olmocr_bench", "doc_id": 17, "repeats": 1, "split": "test"}
    doc = {"image": {"bytes": _png_bytes((10, 20, 30))}}

    request = task.construct_requests(17, "page prompt", metadata=metadata)

    assert request.request_type == "generate_until"
    assert len(request.args) == 6
    assert request.args[0] == "page prompt"
    assert request.args[3:] == (17, "olmocr_bench", "test")
    if protocol == "simple":
        assert request.args[1] == {"temperature": 0.0, "until": []}
        assert request.args[1] is not task.config.generation_kwargs
        assert request.args[2](doc)[0].getpixel((0, 0)) == (10, 20, 30)
    else:
        assert request.args[2] == {"temperature": 0.0, "until": []}
        assert request.args[2] is not task.config.generation_kwargs
        messages = request.args[1](doc)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert [part["type"] for part in messages[0]["content"]] == ["image", "text"]
        assert messages[0]["content"][0]["url"].getpixel((0, 0)) == (10, 20, 30)
        assert messages[0]["content"][1]["text"] == utils.PROMPT_FULL_NO_DOCUMENT_ANCHORING


@pytest.mark.parametrize(
    ("model_name", "protocol"),
    [("dummy", "simple"), ("qwen3_vl", "chat"), ("qwen2_5_vl", "simple"), ("qwen2_5_vl", "chat")],
)
def test_task_manager_loads_yaml_constructor_and_model_protocol(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    protocol: str,
) -> None:
    test = {
        "id": "raw",
        "pdf": "synthetic.pdf",
        "page": 1,
        "type": "present",
        "text": "A",
        "first_n": 1,
        "group": "baseline",
    }
    doc = {
        "pdf": "synthetic.pdf",
        "page": 1,
        "category": "baseline",
        "tests_json": json.dumps([test]),
        "image": {"bytes": _png_bytes((10, 20, 30)), "path": "inline.png"},
    }
    dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list([doc])})
    monkeypatch.setattr(datasets, "load_dataset", lambda **kwargs: dataset)
    manager = TaskManager("ERROR", model_name=model_name)

    task = manager.load_task_or_group("olmocr_bench", task_type=protocol)["olmocr_bench"]

    assert isinstance(task, ConfigurableMessagesTask)
    assert type(task).__name__ == "OlmocrBenchTask"
    assert task.model_name == model_name
    assert task.config.task == "olmocr_bench"
    assert task.config.reasoning_tags is False
    request = task.construct_requests(
        0,
        "page prompt",
        metadata={"task": "olmocr_bench", "doc_id": 0, "repeats": 1, "split": "test"},
    )
    assert request.request_type == "generate_until"
    assert len(request.args) == 6
    assert request.args[0] == "page prompt"
    assert request.args[3:] == (0, "olmocr_bench", "test")
    expected_generation = {"max_new_tokens": 4096, "temperature": 0.0, "do_sample": False, "until": []}
    if protocol == "simple":
        assert request.args[1] == expected_generation
        assert request.args[2](doc)[0].getpixel((0, 0)) == (10, 20, 30)
    else:
        assert request.args[2] == expected_generation
        messages = request.args[1](doc)
        assert messages[0]["role"] == "user"
        assert [part["type"] for part in messages[0]["content"]] == ["image", "text"]
        assert messages[0]["content"][0]["url"].getpixel((0, 0)) == (10, 20, 30)
        assert messages[0]["content"][1]["text"] == utils.PROMPT_FULL_NO_DOCUMENT_ANCHORING
    assert task.process_results(doc, ["  A"])["olmocr_bench_baseline"] == {"passed": 0, "total": 1}


def test_renderer_subprocess_exits_cleanly(integration, tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(utils.__file__).resolve().parents[3])
    script = "from lmms_eval.tasks.olmocr_bench._math import render_equation\n"
    script += "assert render_equation('a+b') is not None\nprint('rendered')"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "rendered"
    assert completed.stderr == ""


def test_custom_task_without_protocol_parameter_keeps_constructor_contract() -> None:
    class ConfigOnlyTask:
        def __init__(self, config, **kwargs):
            self.received_kwargs = kwargs
            self.received_config = config

    manager = TaskManager("ERROR")
    config = {"task": "config_only", "class": ConfigOnlyTask}
    task = manager._load_individual_task_or_group(config, task_type="chat")["config_only"]
    assert task.received_kwargs == {}
    assert task.received_config["task"] == "config_only"

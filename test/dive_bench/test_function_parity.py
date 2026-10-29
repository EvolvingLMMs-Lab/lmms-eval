"""AST hashes independently captured from the pre-port release source.

Whitespace/import relocation is ignored; kernel function bodies are not.
The manifest deliberately excludes four documented, reviewed lint-only deltas
outside the published generation kernels and the new native harness adapters.
"""

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads(Path(__file__).with_name("function_parity.json").read_text())


def function_hashes(path):
    result = {}

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, ast.FunctionDef):
                result[prefix + node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()

    visit(ast.parse(path.read_text()).body)
    return result


@pytest.mark.parametrize("path,entry", MANIFEST.items())
def test_release_function_bodies_unchanged(path, entry):
    current = function_hashes(ROOT / path)
    expected = entry["function_ast_sha256"]
    assert expected
    assert {name: current.get(name) for name in expected} == expected

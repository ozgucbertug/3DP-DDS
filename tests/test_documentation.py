from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def test_readme_python_blocks_parse() -> None:
    contents = README.read_text(encoding="utf-8")
    blocks = PYTHON_BLOCK.findall(contents)

    assert blocks
    for index, block in enumerate(blocks, start=1):
        ast.parse(block, filename=f"README.md python block {index}")


def test_readme_does_not_reference_removed_convenience_apis() -> None:
    contents = README.read_text(encoding="utf-8")

    assert "bundle.strata(" not in contents
    assert "bundle.interface(" not in contents
    assert "bundle.support(" not in contents
    assert "result.show(" not in contents
    assert "sim.show(" not in contents

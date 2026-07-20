from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_files_parse_with_python39_grammar() -> None:
    for path in sorted((ROOT / "src").rglob("*.py")):
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path.relative_to(ROOT)),
            feature_version=(3, 9),
        )

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "source"
OUTPUT_DIR = ROOT / "docs"
DOCTREE_DIR = ROOT / "docs" / "_build" / "doctrees"

GENERATED_PATHS = (
    ".buildinfo",
    ".doctrees",
    ".nojekyll",
    "_build",
    "_modules",
    "_sources",
    "_static",
    "api",
    "concepts",
    "genindex.html",
    "getting-started.html",
    "index.html",
    "objects.inv",
    "search.html",
    "searchindex.js",
    "tutorials",
)


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def require_clean_tree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        raise SystemExit("Working tree must be clean before rebuilding docs.")


def clean_generated_docs() -> None:
    for relative_path in GENERATED_PATHS:
        path = OUTPUT_DIR / relative_path
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> int:
    original_branch = output("git", "branch", "--show-current")
    if original_branch != "main":
        raise SystemExit("Run scripts/publish_docs.py from the main branch.")

    require_clean_tree()

    clean_generated_docs()
    if DOCTREE_DIR.exists():
        shutil.rmtree(DOCTREE_DIR)

    run(
        "sphinx-build",
        "-W",
        "--keep-going",
        "-b",
        "html",
        "-d",
        str(DOCTREE_DIR),
        str(SOURCE_DIR),
        str(OUTPUT_DIR),
    )
    if not (OUTPUT_DIR / ".nojekyll").exists():
        raise SystemExit("Sphinx build did not create .nojekyll.")
    if not (OUTPUT_DIR / "index.html").exists():
        raise SystemExit("Sphinx build did not create index.html.")

    # Sphinx creates this directory even with html_copy_source disabled.
    sources_dir = OUTPUT_DIR / "_sources"
    if sources_dir.exists():
        shutil.rmtree(sources_dir)

    print("Built GitHub Pages site in docs/. Commit docs/ after reviewing the generated output.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

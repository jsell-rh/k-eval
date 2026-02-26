"""Implementation of the `k-eval view` command.

Reads a .detailed.jsonl results file, inlines the data into the single-file
HTML viewer, writes a temp file, and opens it in the default browser.
"""

import importlib.resources
import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

type JsonRecord = dict[str, Any]

_PLACEHOLDER = "window.__KEVAL_DATA__ = null;"


def _load_template() -> str:
    """Return the viewer HTML template as a string."""
    pkg_files = importlib.resources.files("k_eval.viewer")
    return pkg_files.joinpath("viewer.html").read_text(encoding="utf-8")


def build_viewer_html(records: list[JsonRecord]) -> str:
    """Inline *records* into the viewer HTML template and return the result.

    The template contains a JavaScript placeholder::

        window.__KEVAL_DATA__ = null;

    which is replaced with the serialised JSON array so that the page loads
    data immediately without requiring a file-load interaction.
    """
    template = _load_template()
    data_json = json.dumps(records)
    replacement = f"window.__KEVAL_DATA__ = {data_json};"
    return template.replace(_PLACEHOLDER, replacement, 1)


def open_viewer(jsonl_path: Path) -> None:
    """Open *jsonl_path* in the browser via a temporary HTML file.

    Raises:
        FileNotFoundError: if *jsonl_path* does not exist.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Failed to open viewer: results file not found: {jsonl_path}"
        )

    raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    records: list[JsonRecord] = [json.loads(line) for line in raw_lines if line.strip()]

    html = build_viewer_html(records=records)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        prefix="keval_viewer_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(html)
        tmp_path = Path(tmp.name)

    url = f"file://{tmp_path}"
    print(f"Opened: {tmp_path}")
    webbrowser.open(url)

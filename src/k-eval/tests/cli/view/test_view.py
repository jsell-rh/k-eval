"""Tests for cli/view/command.py — the `k-eval view` command logic."""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from k_eval.cli.view.command import build_viewer_html, open_viewer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(sample_idx: str = "0", condition: str = "baseline") -> dict[str, Any]:
    return {
        "schema_version": "instance_level_eval_0.2.1",
        "evaluation_id": "test-run-id",
        "model_id": "claude-test",
        "evaluation_name": f"test-eval/{condition}",
        "sample_idx": sample_idx,
        "interaction_type": "agentic",
        "input": {
            "raw": f"Question {sample_idx}?",
            "reference": f"Answer {sample_idx}.",
        },
        "output": {"raw_runs": ["Agent response."]},
        "evaluation": {
            "score": None,
            "details": {
                "factual_adherence_mean": 4.0,
                "factual_adherence_stddev": 0.5,
                "factual_adherence_reasonings": ["Reasoning FA."],
                "completeness_mean": 3.5,
                "completeness_stddev": 0.2,
                "completeness_reasonings": ["Reasoning CO."],
                "helpfulness_and_clarity_mean": 5.0,
                "helpfulness_and_clarity_stddev": 0.0,
                "helpfulness_and_clarity_reasonings": ["Reasoning HC."],
                "unverified_claims": ["Some claim."],
            },
        },
        "token_usage": {"input_tokens": 100, "output_tokens": 50},
        "run_details": [
            {
                "repetition_index": 0,
                "agent_response": "Agent response.",
                "cost_usd": 0.01,
                "duration_ms": 5000,
                "num_turns": 2,
            }
        ],
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# build_viewer_html
# ---------------------------------------------------------------------------


class TestBuildViewerHtml:
    """build_viewer_html inlines data into the HTML template."""

    def test_returns_string(self) -> None:
        records = [_make_record()]
        result = build_viewer_html(records=records)
        assert isinstance(result, str)

    def test_contains_inlined_data(self) -> None:
        records = [_make_record(sample_idx="42")]
        result = build_viewer_html(records=records)
        assert '"sample_idx": "42"' in result or '"42"' in result

    def test_data_placeholder_is_replaced(self) -> None:
        records = [_make_record()]
        result = build_viewer_html(records=records)
        # The null placeholder must not remain
        assert "window.__KEVAL_DATA__ = null;" not in result

    def test_inlined_data_is_valid_json_array(self) -> None:
        records = [_make_record(sample_idx="0"), _make_record(sample_idx="1")]
        result = build_viewer_html(records=records)
        # Extract the JSON array from the injected assignment
        marker = "window.__KEVAL_DATA__ = "
        start = result.index(marker) + len(marker)
        end = result.index(";\n", start)
        parsed = json.loads(result[start:end])
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_html_contains_doctype(self) -> None:
        result = build_viewer_html(records=[_make_record()])
        assert result.strip().startswith("<!DOCTYPE html>")

    def test_empty_records_list_inlines_empty_array(self) -> None:
        result = build_viewer_html(records=[])
        assert "window.__KEVAL_DATA__ = [];" in result


# ---------------------------------------------------------------------------
# open_viewer
# ---------------------------------------------------------------------------


class TestOpenViewer:
    """open_viewer writes a temp file and calls webbrowser.open."""

    def test_opens_browser_with_file_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.detailed.jsonl"
            _write_jsonl(jsonl_path, [_make_record()])

            with patch("k_eval.cli.view.command.webbrowser.open") as mock_open:
                open_viewer(jsonl_path=jsonl_path)

            mock_open.assert_called_once()
            url: str = mock_open.call_args[0][0]
            assert url.startswith("file://")
            assert url.endswith(".html")

    def test_printed_path_is_the_temp_file(
        self, capsys: pytest.CaptureFixture[Any]
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.detailed.jsonl"
            _write_jsonl(jsonl_path, [_make_record()])

            with patch("k_eval.cli.view.command.webbrowser.open"):
                open_viewer(jsonl_path=jsonl_path)

            captured = capsys.readouterr()
            assert "Opened:" in captured.out

    def test_temp_html_file_contains_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.detailed.jsonl"
            records = [_make_record(sample_idx="99")]
            _write_jsonl(jsonl_path, records)

            written_html: list[str] = []

            def capture_open(url: str) -> None:
                # url is file:///path/to/file.html
                path = Path(url[len("file://") :])
                written_html.append(path.read_text(encoding="utf-8"))

            with patch(
                "k_eval.cli.view.command.webbrowser.open", side_effect=capture_open
            ):
                open_viewer(jsonl_path=jsonl_path)

            assert len(written_html) == 1
            assert '"sample_idx": "99"' in written_html[0] or "'99'" in written_html[0]

    def test_raises_for_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            open_viewer(jsonl_path=Path("/nonexistent/path/results.jsonl"))

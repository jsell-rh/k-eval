#!/usr/bin/env python3
"""Merge two k-eval *.detailed.jsonl runs into one file comparing ask_sre variants.

Example (from repo root):

  uv run python scripts/merge_ask_sre_detailed.py \\
    --current results/test1_20260316_a2fc8f7b.detailed.jsonl \\
    --enhanced results/test1_20260326_eb517976.detailed.jsonl \\
    -o results/test1_ask_sre_current_vs_enhanced.detailed.jsonl

Expects both inputs to contain the same sample_idx set and matching questions
(51-row ask_sre runs). Strips non-ask_sre conditions from the current file if
present (e.g. kartograph).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

type JsonObj = dict[str, Any]


def _load_jsonl(path: Path) -> list[JsonObj]:
    lines: list[JsonObj] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines


def _config_prefix(evaluation_name: str) -> str:
    parts = evaluation_name.split("/", 1)
    if len(parts) != 2:
        msg = f"Failed to parse evaluation_name (expected config/condition): {evaluation_name!r}"
        raise ValueError(msg)
    return parts[0]


def _filter_ask_sre(records: list[JsonObj], source_name: str) -> list[JsonObj]:
    out = [r for r in records if r.get("evaluation_name") == source_name]
    if not out:
        msg = f"No rows with evaluation_name={source_name!r} in {records[0] if records else 'empty file'}"
        raise SystemExit(f"Failed to merge: {msg}")
    return out


def _validate_alignment(current: list[JsonObj], enhanced: list[JsonObj]) -> None:
    by_c = {str(r["sample_idx"]): r for r in current}
    by_e = {str(r["sample_idx"]): r for r in enhanced}
    keys_c, keys_e = set(by_c), set(by_e)
    if keys_c != keys_e:
        only_c = sorted(keys_c - keys_e)
        only_e = sorted(keys_e - keys_c)
        raise SystemExit(
            "Failed to merge: sample_idx sets differ.\n"
            f"  Only in current: {only_c[:10]}{'...' if len(only_c) > 10 else ''}\n"
            f"  Only in enhanced: {only_e[:10]}{'...' if len(only_e) > 10 else ''}"
        )
    mismatches: list[str] = []
    for sid in sorted(keys_c, key=lambda x: int(x) if x.isdigit() else 0):
        qc = by_c[sid]["input"]["raw"]
        qe = by_e[sid]["input"]["raw"]
        if qc != qe:
            mismatches.append(f"sample_idx={sid!r}")
    if mismatches:
        raise SystemExit(
            "Failed to merge: question text differs for: " + ", ".join(mismatches[:5])
            + (" ..." if len(mismatches) > 5 else "")
        )


def _rewrite(
    records: list[JsonObj],
    *,
    evaluation_name: str,
    evaluation_id: str,
) -> list[JsonObj]:
    out: list[JsonObj] = []
    for r in records:
        row = dict(r)
        row["evaluation_name"] = evaluation_name
        row["evaluation_id"] = evaluation_id
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Older .detailed.jsonl (ask_sre rows will be labeled ask_sre current)",
    )
    p.add_argument(
        "--enhanced",
        type=Path,
        required=True,
        help="Newer .detailed.jsonl (ask_sre rows labeled ask_sre enhanced)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Merged .detailed.jsonl path",
    )
    p.add_argument(
        "--source-name",
        default="test1/ask_sre",
        help="evaluation_name to keep from both files (default: test1/ask_sre)",
    )
    p.add_argument(
        "--current-label",
        default="ask_sre current",
        help="Condition label after config/ (default: ask_sre current)",
    )
    p.add_argument(
        "--enhanced-label",
        default="ask_sre enhanced",
        help="Condition label after config/ (default: ask_sre enhanced)",
    )
    p.add_argument(
        "--merged-id",
        default=None,
        help="evaluation_id for all output rows (default: merged_<current_stem>_<enhanced_stem>)",
    )
    args = p.parse_args()

    cur_all = _load_jsonl(args.current)
    enh_all = _load_jsonl(args.enhanced)
    cur = _filter_ask_sre(cur_all, args.source_name)
    enh = _filter_ask_sre(enh_all, args.source_name)
    _validate_alignment(cur, enh)

    prefix = _config_prefix(cur[0]["evaluation_name"])
    name_current = f"{prefix}/{args.current_label}"
    name_enhanced = f"{prefix}/{args.enhanced_label}"

    merged_id = args.merged_id
    if merged_id is None:
        merged_id = f"merged_{args.current.stem}_{args.enhanced.stem}"

    merged = _rewrite(cur, evaluation_name=name_current, evaluation_id=merged_id) + _rewrite(
        enh, evaluation_name=name_enhanced, evaluation_id=merged_id
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(merged)} lines to {args.output}\n"
        f"  {name_current}: {len(cur)} samples\n"
        f"  {name_enhanced}: {len(enh)}\n"
        f"  evaluation_id: {merged_id}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REQUIRED_KEYS = ("uid", "clean", "corrupt", "clean_p_yes", "corrupt_p_yes")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths() -> dict[str, Path]:
    base = repo_root() / "dataset" / "semantic_meaningful"
    return {
        "filtered": base / "filtered_pairs_prefix_local.jsonl",
        "rejected": base / "rejected_pairs_prefix_local.jsonl",
        "merged": base / "merged_scored_pairs_prefix_local.jsonl",
        "summary": base / "merged_scored_pairs_prefix_summary_local.json",
    }


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Merge filtered and rejected prefix-pair JSONL files into a single "
            "scored dataset for post-hoc threshold tuning."
        )
    )
    parser.add_argument("--input-filtered", type=Path, default=paths["filtered"])
    parser.add_argument("--input-rejected", type=Path, default=paths["rejected"])
    parser.add_argument("--output-merged", type=Path, default=paths["merged"])
    parser.add_argument("--output-summary", type=Path, default=paths["summary"])
    parser.add_argument(
        "--sort-by-uid",
        action="store_true",
        help="Sort merged output deterministically by uid before writing.",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail on malformed rows or duplicate UIDs (default).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip malformed rows and duplicate UIDs instead of failing.",
    )
    return parser.parse_args()


def _normalize_row(row: dict, source: str) -> dict:
    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        raise ValueError(f"Missing required keys {missing}")

    normalized = dict(row)
    normalized["uid"] = str(row["uid"])
    normalized["clean_p_yes"] = float(row["clean_p_yes"])
    normalized["corrupt_p_yes"] = float(row["corrupt_p_yes"])
    normalized["original_split"] = source
    return normalized


def main() -> None:
    args = parse_args()

    for required_path in [args.input_filtered, args.input_rejected]:
        if not required_path.exists():
            raise FileNotFoundError(f"Input file not found: {required_path}")

    filtered_rows = load_jsonl(args.input_filtered)
    rejected_rows = load_jsonl(args.input_rejected)

    merged_rows: list[dict] = []
    seen_uids: set[str] = set()
    duplicate_uids: list[str] = []
    malformed_rows = 0
    malformed_messages: list[str] = []

    for source_name, rows in (("filtered", filtered_rows), ("rejected", rejected_rows)):
        for idx, row in enumerate(rows, start=1):
            try:
                normalized = _normalize_row(row, source=source_name)
            except Exception as exc:  # pragma: no cover - defensive input handling
                malformed_rows += 1
                if len(malformed_messages) < 10:
                    malformed_messages.append(f"{source_name} row {idx}: {exc}")
                if args.strict:
                    continue
                continue

            uid = normalized["uid"]
            if uid in seen_uids:
                duplicate_uids.append(uid)
                if args.strict:
                    continue
                continue

            seen_uids.add(uid)
            merged_rows.append(normalized)

    error_chunks: list[str] = []
    if malformed_rows > 0:
        error_chunks.append(f"malformed_rows={malformed_rows}")
    if duplicate_uids:
        error_chunks.append(f"duplicate_uids={len(duplicate_uids)}")

    if args.strict and error_chunks:
        sample_dupes = duplicate_uids[:5]
        sample_msg = "; ".join(malformed_messages)
        raise ValueError(
            "Strict mode validation failed: "
            + ", ".join(error_chunks)
            + (f" | duplicate uid examples: {sample_dupes}" if sample_dupes else "")
            + (f" | malformed examples: {sample_msg}" if sample_msg else "")
        )

    if args.sort_by_uid:
        merged_rows.sort(key=lambda x: x["uid"])

    written_rows = write_jsonl(args.output_merged, merged_rows)

    summary = {
        "meta": {
            "script": "merge_prefix_scored_pairs.py",
            "description": "Merge filtered/rejected prefix scoring outputs into one scored JSONL.",
            "strict": bool(args.strict),
            "sort_by_uid": bool(args.sort_by_uid),
        },
        "counts": {
            "filtered_input_rows": len(filtered_rows),
            "rejected_input_rows": len(rejected_rows),
            "merged_rows_written": written_rows,
            "malformed_rows": malformed_rows,
            "duplicate_uids": len(duplicate_uids),
        },
        "samples": {
            "duplicate_uids": duplicate_uids[:20],
            "malformed_messages": malformed_messages,
        },
        "paths": {
            "input_filtered": str(args.input_filtered),
            "input_rejected": str(args.input_rejected),
            "output_merged": str(args.output_merged),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Filtered rows in: {len(filtered_rows):,}")
    print(f"Rejected rows in: {len(rejected_rows):,}")
    print(f"Merged rows out: {written_rows:,}")
    print(f"Malformed rows skipped: {malformed_rows:,}")
    print(f"Duplicate UIDs skipped: {len(duplicate_uids):,}")
    print(f"Merged output: {args.output_merged.resolve()}")
    print(f"Summary output: {args.output_summary.resolve()}")


if __name__ == "__main__":
    main()
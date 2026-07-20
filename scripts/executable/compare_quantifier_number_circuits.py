from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DEFAULT_TOP_K = (30, 50, 100, 200, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two quantifier-number discovery circuits by edge-set overlap."
    )
    parser.add_argument("--left-summary", type=Path, required=True)
    parser.add_argument("--right-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K))
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def edge_path_from_summary(summary: dict[str, Any]) -> Path:
    edge_path = summary.get("paths", {}).get("edge_rankings")
    if not edge_path:
        raise ValueError("Summary does not contain paths.edge_rankings")
    return Path(edge_path)


def reference_budget_from_summary(summary: dict[str, Any]) -> int:
    source = summary.get("overlap", {}).get("quantifier_number_source", {})
    reference_budget = source.get("reference_budget")
    if reference_budget is None:
        raise ValueError("Summary does not contain overlap.quantifier_number_source.reference_budget")
    return int(reference_budget)


def edge_names(edge_path: Path, budget: int | None = None) -> list[str]:
    frame = pd.read_csv(edge_path)
    if "collapsed_edge" not in frame.columns:
        raise ValueError(f"Edge ranking file is missing 'collapsed_edge': {edge_path}")
    values = frame["collapsed_edge"].astype(str).tolist()
    return values if budget is None else values[:budget]


def overlap(
    left_edges: Sequence[str],
    right_edges: Sequence[str],
    *,
    left_budget: int,
    right_budget: int,
) -> dict[str, Any]:
    left = set(left_edges[:left_budget])
    right = set(right_edges[:right_budget])
    intersection = left & right
    union = left | right
    return {
        "left_budget": int(left_budget),
        "right_budget": int(right_budget),
        "left_count": int(len(left)),
        "right_count": int(len(right)),
        "overlap_count": int(len(intersection)),
        "left_overlap_rate": float(len(intersection) / len(left)) if left else 0.0,
        "right_overlap_rate": float(len(intersection) / len(right)) if right else 0.0,
        "jaccard": float(len(intersection) / len(union)) if union else 0.0,
        "overlap_edges": sorted(intersection),
    }


def main() -> None:
    args = parse_args()
    left_summary = load_summary(args.left_summary)
    right_summary = load_summary(args.right_summary)
    left_edge_path = edge_path_from_summary(left_summary)
    right_edge_path = edge_path_from_summary(right_summary)
    left_edges = edge_names(left_edge_path)
    right_edges = edge_names(right_edge_path)
    left_reference_budget = reference_budget_from_summary(left_summary)
    right_reference_budget = reference_budget_from_summary(right_summary)

    top_k = {
        f"top_{budget}": overlap(
            left_edges,
            right_edges,
            left_budget=budget,
            right_budget=budget,
        )
        for budget in args.top_k
        if budget <= len(left_edges) and budget <= len(right_edges)
    }
    payload = {
        "left": {
            "summary": str(args.left_summary),
            "edge_path": str(left_edge_path),
            "target_class": left_summary.get("target_class"),
            "reference_budget": left_reference_budget,
        },
        "right": {
            "summary": str(args.right_summary),
            "edge_path": str(right_edge_path),
            "target_class": right_summary.get("target_class"),
            "reference_budget": right_reference_budget,
        },
        "reference_overlap": overlap(
            left_edges,
            right_edges,
            left_budget=left_reference_budget,
            right_budget=right_reference_budget,
        ),
        "same_left_reference_budget": overlap(
            left_edges,
            right_edges,
            left_budget=left_reference_budget,
            right_budget=min(left_reference_budget, len(right_edges)),
        ),
        "same_rank_overlap": top_k,
    }
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

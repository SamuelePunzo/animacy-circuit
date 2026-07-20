from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


EDGE_FILE_RE = re.compile(r"edges_sample_(?P<sample_size>\d+)_seed_(?P<seed>\d+)\.csv$")


@dataclass(frozen=True)
class RunSlot:
    model_slug: str
    run_name: str
    sample_size: int
    seed: int
    edge_path: Path
    eval_path: Path | None
    partial_eval_path: Path | None


def split_underlying_edges(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    return [part for part in str(value).split("|") if part]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def is_main_original_run(model_slug: str, run_name: str) -> bool:
    del model_slug
    text = run_name.lower()
    excluded_tokens = ("probe", "smoke", "named_entity", "control", "shadow", "import")
    return "seed_stability" in text and not any(token in text for token in excluded_tokens)


def find_run_slots(
    results_root: Path,
    model_queries: list[str] | None = None,
    *,
    main_original_only: bool = False,
) -> list[RunSlot]:
    if not results_root.exists():
        raise FileNotFoundError(f"Localization results root does not exist: {results_root}")

    query_terms = [query.lower() for query in model_queries or []]
    slots: list[RunSlot] = []
    for edge_path in sorted(results_root.glob("*/*/sample_*/seed_*/edges_sample_*_seed_*.csv")):
        match = EDGE_FILE_RE.match(edge_path.name)
        if match is None:
            continue
        model_slug = edge_path.relative_to(results_root).parts[0]
        run_name = edge_path.relative_to(results_root).parts[1]
        if main_original_only and not is_main_original_run(model_slug, run_name):
            continue
        haystack = f"{model_slug}/{run_name}".lower()
        if query_terms and not any(query in haystack for query in query_terms):
            continue

        sample_size = int(match.group("sample_size"))
        seed = int(match.group("seed"))
        eval_name = f"topk_evaluations_sample_{sample_size}_seed_{seed}.csv"
        partial_eval_name = f"topk_evaluations_partial_sample_{sample_size}_seed_{seed}.csv"
        eval_path = edge_path.with_name(eval_name)
        partial_eval_path = edge_path.with_name(partial_eval_name)
        slots.append(
            RunSlot(
                model_slug=model_slug,
                run_name=run_name,
                sample_size=sample_size,
                seed=seed,
                edge_path=edge_path,
                eval_path=eval_path if eval_path.exists() else None,
                partial_eval_path=partial_eval_path if partial_eval_path.exists() else None,
            )
        )
    return slots


def ranked_ablation_rows(evaluations: pd.DataFrame) -> pd.DataFrame:
    required = {"mode", "collapsed_edge_budget", "faithfulness_mean"}
    missing = sorted(required - set(evaluations.columns))
    if missing:
        raise ValueError(f"Top-k evaluation frame is missing required columns: {missing}")

    rows = evaluations[evaluations["mode"].astype(str) == "ablate_top"].copy()
    if "baseline" in rows.columns:
        rows = rows[rows["baseline"].astype(str) == "eap_ranked"]
    if "matched_random" in rows.columns:
        rows = rows[~rows["matched_random"].apply(parse_bool)]
    if "repeat" in rows.columns:
        rows = rows[rows["repeat"].fillna(0).astype(int) == 0]

    rows["collapsed_edge_budget"] = rows["collapsed_edge_budget"].astype(int)
    rows["faithfulness_mean"] = rows["faithfulness_mean"].astype(float)
    return rows.sort_values("collapsed_edge_budget", kind="stable").reset_index(drop=True)


def first_budget_below_faithfulness(
    evaluations: pd.DataFrame,
    threshold: float,
    candidate_budgets: set[int] | None = None,
) -> dict[str, Any] | None:
    rows = ranked_ablation_rows(evaluations)
    if candidate_budgets is not None:
        rows = rows[rows["collapsed_edge_budget"].isin(candidate_budgets)].copy()
    hits = rows[rows["faithfulness_mean"] < float(threshold)]
    if hits.empty:
        return None
    row = hits.iloc[0].to_dict()
    row["collapsed_edge_budget"] = int(row["collapsed_edge_budget"])
    row["faithfulness_mean"] = float(row["faithfulness_mean"])
    if "accuracy_mean" in row and pd.notna(row["accuracy_mean"]):
        row["accuracy_mean"] = float(row["accuracy_mean"])
    return row


def load_selected_edges(edge_path: Path, budget: int) -> pd.DataFrame:
    edges = pd.read_csv(edge_path)
    required = {"collapsed_edge", "parent", "child", "abs_score", "underlying_edges"}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise ValueError(f"Edge ranking {edge_path} is missing required columns: {missing}")

    if "rank" in edges.columns:
        ordered = edges.sort_values("rank", ascending=True, kind="stable").reset_index(drop=True)
    else:
        ordered = edges.sort_values("abs_score", ascending=False, kind="stable").reset_index(drop=True)
        ordered.insert(0, "rank", range(1, len(ordered) + 1))

    if budget > len(ordered):
        raise ValueError(f"Selected budget {budget} exceeds available collapsed edges {len(ordered)} in {edge_path}")

    selected = ordered.head(budget).copy()
    selected["underlying_edge_list"] = selected["underlying_edges"].apply(split_underlying_edges)
    if "underlying_edge_count" not in selected.columns:
        selected["underlying_edge_count"] = selected["underlying_edge_list"].apply(len)
    return selected


def expand_underlying_edges(selected_edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for collapsed_index, edge in selected_edges.reset_index(drop=True).iterrows():
        underlying = edge["underlying_edge_list"]
        for underlying_index, edge_name in enumerate(underlying, start=1):
            rows.append(
                {
                    "collapsed_edge_index": int(collapsed_index + 1),
                    "underlying_edge_index": int(underlying_index),
                    "underlying_edge": str(edge_name),
                    "collapsed_edge": str(edge["collapsed_edge"]),
                    "collapsed_rank": int(edge["rank"]),
                    "parent": str(edge["parent"]),
                    "child": str(edge["child"]),
                    "collapsed_abs_score": float(edge["abs_score"]),
                    "collapsed_signed_sum": float(edge.get("signed_sum", edge["abs_score"])),
                    "collapsed_underlying_edge_count": int(edge["underlying_edge_count"]),
                }
            )
    return pd.DataFrame(rows)


def analyze_slot(
    slot: RunSlot,
    *,
    threshold: float,
    allow_partial: bool,
    candidate_budgets: set[int] | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    eval_path = slot.eval_path
    source_kind = "complete"
    if eval_path is None:
        if slot.partial_eval_path is None:
            return (
                summary_row(slot, threshold=threshold, status="missing_eval"),
                pd.DataFrame(),
                pd.DataFrame(),
            )
        if not allow_partial:
            return (
                summary_row(slot, threshold=threshold, status="incomplete_topk"),
                pd.DataFrame(),
                pd.DataFrame(),
            )
        eval_path = slot.partial_eval_path
        source_kind = "partial"

    try:
        evaluations = pd.read_csv(eval_path)
        selected_row = first_budget_below_faithfulness(evaluations, threshold, candidate_budgets)
    except Exception as exc:
        return (
            summary_row(slot, threshold=threshold, status="invalid_eval", error=str(exc), eval_source=source_kind),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if selected_row is None:
        ablation_rows = ranked_ablation_rows(evaluations)
        if candidate_budgets is not None:
            ablation_rows = ablation_rows[ablation_rows["collapsed_edge_budget"].isin(candidate_budgets)].copy()
        max_budget = int(ablation_rows["collapsed_edge_budget"].max()) if not ablation_rows.empty else None
        min_faithfulness = float(ablation_rows["faithfulness_mean"].min()) if not ablation_rows.empty else None
        return (
            summary_row(
                slot,
                threshold=threshold,
                status="no_candidate_budget_hit" if candidate_budgets is not None else "no_threshold_hit",
                eval_source=source_kind,
                max_evaluated_budget=max_budget,
                min_ablation_faithfulness=min_faithfulness,
                candidate_budgets="|".join(str(budget) for budget in sorted(candidate_budgets))
                if candidate_budgets is not None
                else None,
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    budget = int(selected_row["collapsed_edge_budget"])
    try:
        selected_edges = load_selected_edges(slot.edge_path, budget)
        expanded_edges = expand_underlying_edges(selected_edges)
    except Exception as exc:
        return (
            summary_row(
                slot,
                threshold=threshold,
                status="invalid_edges",
                error=str(exc),
                selected_budget=budget,
                selected_faithfulness=float(selected_row["faithfulness_mean"]),
                eval_source=source_kind,
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    metadata = slot_metadata(slot, threshold=threshold)
    selected_edges = selected_edges.copy()
    selected_edges.insert(0, "model_slug", slot.model_slug)
    selected_edges.insert(1, "run_name", slot.run_name)
    selected_edges.insert(2, "sample_size", slot.sample_size)
    selected_edges.insert(3, "seed", slot.seed)
    selected_edges.insert(4, "selected_budget", budget)
    selected_edges.insert(5, "selected_faithfulness_mean", float(selected_row["faithfulness_mean"]))
    if "accuracy_mean" in selected_row:
        selected_edges.insert(6, "selected_accuracy_mean", selected_row.get("accuracy_mean"))
    selected_edges = selected_edges.drop(columns=["underlying_edge_list"])

    expanded_edges = expanded_edges.copy()
    for key, value in reversed(metadata.items()):
        expanded_edges.insert(0, key, value)
    expanded_edges.insert(4, "selected_budget", budget)
    expanded_edges.insert(5, "selected_faithfulness_mean", float(selected_row["faithfulness_mean"]))
    if "accuracy_mean" in selected_row:
        expanded_edges.insert(6, "selected_accuracy_mean", selected_row.get("accuracy_mean"))

    summary = summary_row(
        slot,
        threshold=threshold,
        status="selected" if source_kind == "complete" else "selected_from_partial",
        eval_source=source_kind,
        selected_budget=budget,
        selected_faithfulness=float(selected_row["faithfulness_mean"]),
        selected_accuracy=selected_row.get("accuracy_mean"),
        selected_collapsed_edge_count=int(len(selected_edges)),
        selected_underlying_edge_count=int(len(expanded_edges)),
    )
    return summary, selected_edges, expanded_edges


def slot_metadata(slot: RunSlot, *, threshold: float) -> dict[str, Any]:
    return {
        "model_slug": slot.model_slug,
        "run_name": slot.run_name,
        "sample_size": int(slot.sample_size),
        "seed": int(slot.seed),
        "threshold": float(threshold),
    }


def summary_row(slot: RunSlot, *, threshold: float, status: str, **extra: Any) -> dict[str, Any]:
    row = {
        **slot_metadata(slot, threshold=threshold),
        "status": status,
        "edge_path": str(slot.edge_path),
        "eval_path": str(slot.eval_path) if slot.eval_path is not None else None,
        "partial_eval_path": str(slot.partial_eval_path) if slot.partial_eval_path is not None else None,
    }
    row.update(extra)
    return row


def run_analysis(
    *,
    results_root: Path,
    output_dir: Path,
    threshold: float,
    model_queries: list[str] | None = None,
    allow_partial: bool = False,
    candidate_budgets: list[int] | None = None,
    main_original_only: bool = False,
) -> dict[str, Any]:
    candidate_budget_set = {int(budget) for budget in candidate_budgets} if candidate_budgets is not None else None
    slots = find_run_slots(
        results_root,
        model_queries=model_queries,
        main_original_only=main_original_only,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    collapsed_frames: list[pd.DataFrame] = []
    underlying_frames: list[pd.DataFrame] = []
    for slot in slots:
        summary, collapsed, underlying = analyze_slot(
            slot,
            threshold=threshold,
            allow_partial=allow_partial,
            candidate_budgets=candidate_budget_set,
        )
        summary_rows.append(summary)
        if not collapsed.empty:
            collapsed_frames.append(collapsed)
        if not underlying.empty:
            underlying_frames.append(underlying)

    summary_frame = pd.DataFrame(summary_rows)
    collapsed_frame = pd.concat(collapsed_frames, ignore_index=True) if collapsed_frames else pd.DataFrame()
    underlying_frame = pd.concat(underlying_frames, ignore_index=True) if underlying_frames else pd.DataFrame()

    summary_path = output_dir / "necessary_budget_summary.csv"
    collapsed_path = output_dir / "necessary_collapsed_edges.csv"
    underlying_path = output_dir / "necessary_underlying_edges.csv"
    manifest_path = output_dir / "necessary_budget_summary.json"

    summary_frame.to_csv(summary_path, index=False)
    collapsed_frame.to_csv(collapsed_path, index=False)
    underlying_frame.to_csv(underlying_path, index=False)

    manifest = {
        "experiment": "necessary_collapsed_edge_expansion",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": float(threshold),
        "allow_partial": bool(allow_partial),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "model_queries": model_queries or [],
        "candidate_budgets": sorted(candidate_budget_set) if candidate_budget_set is not None else None,
        "main_original_only": bool(main_original_only),
        "slot_count": int(len(slots)),
        "selected_slot_count": int((summary_frame["status"] == "selected").sum()) if not summary_frame.empty else 0,
        "paths": {
            "summary": str(summary_path),
            "collapsed_edges": str(collapsed_path),
            "underlying_edges": str(underlying_path),
            "manifest": str(manifest_path),
        },
        "status_counts": summary_frame["status"].value_counts().to_dict() if not summary_frame.empty else {},
        "slots": [asdict(slot) | {"edge_path": str(slot.edge_path), "eval_path": str(slot.eval_path) if slot.eval_path else None, "partial_eval_path": str(slot.partial_eval_path) if slot.partial_eval_path else None} for slot in slots],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def default_results_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    if (root / "animacy-circuit").is_dir():
        root = root / "animacy-circuit"
    return root / "results" / "eap_ig_localization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select per-run necessary collapsed edges from localization ablation diagnostics "
            "and expand them into underlying EAP edges."
        )
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--candidate-budgets",
        type=int,
        nargs="+",
        default=None,
        help="Restrict threshold selection to these collapsed-edge budgets, e.g. 20 50.",
    )
    parser.add_argument(
        "--model-query",
        action="append",
        default=None,
        help="Substring filter over model/run path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use topk_evaluations_partial files when complete top-k evaluations are unavailable.",
    )
    parser.add_argument(
        "--main-original-only",
        action="store_true",
        help="Keep only original main seed-stability localization runs; exclude probes, smoke, named-entity, and controls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root or default_results_root()
    day = datetime.now().strftime("%Y-%m-%d")
    output_dir = args.output_dir or (results_root / f"necessary_edge_expansion_{day}")
    manifest = run_analysis(
        results_root=results_root,
        output_dir=output_dir,
        threshold=args.threshold,
        model_queries=args.model_query,
        allow_partial=args.allow_partial,
        candidate_budgets=args.candidate_budgets,
        main_original_only=args.main_original_only,
    )
    print(f"Saved necessary edge expansion reports to {manifest['output_dir']}")
    print(f"Summary: {manifest['paths']['summary']}")
    print(f"Collapsed edges: {manifest['paths']['collapsed_edges']}")
    print(f"Underlying edges: {manifest['paths']['underlying_edges']}")


if __name__ == "__main__":
    main()

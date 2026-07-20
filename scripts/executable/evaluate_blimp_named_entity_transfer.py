from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circuit_finder_core import (  # noqa: E402
    MODEL_SPECIFIC_CORRECT,
    build_budget_circuit,
    build_graph,
    canonical_model_name,
    first_budget_reaching_faithfulness,
    load_model,
    parse_ranked_edge_frame,
    resolve_animacy_circuit_root,
    safe_model_name,
    save_csv,
    save_json,
    target_source_slug,
    token_count_no_special,
)
from control_runners import (  # noqa: E402
    evaluate_blimp_passive_prefix_control,
    load_local_blimp_prefix_dataset,
    summarize_blimp_passive_prefix_control,
)
from evaluate_named_entity_circuit import (  # noqa: E402
    named_entity_target_tensors,
    resolve_valid_source_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing animacy and named-entity circuits on BLiMP "
            "animate_subject_passive prefixes truncated from '... by some/the' to '... by' "
            "using named-entity target sets."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset-set", default=MODEL_SPECIFIC_CORRECT)
    parser.add_argument(
        "--target-source",
        default="dataset/semantic_meaningful/named_entity_targets.json",
        help="Named-entity target JSON path or alias.",
    )
    parser.add_argument(
        "--target-token-mode",
        choices=("first_token", "whole_entity_single_token"),
        default="first_token",
    )
    parser.add_argument(
        "--blimp-config",
        default="animate_subject_passive",
        help="Local BLiMP JSONL stem under dataset/blimp/.",
    )
    parser.add_argument(
        "--original-main-experiment-path",
        default=None,
        help="Original/common-noun full_model directory or artifact path.",
    )
    parser.add_argument(
        "--named-entity-summary-path",
        default=None,
        help="Named-entity discovery summary JSON. Defaults to latest matching run.",
    )
    parser.add_argument("--source-faithfulness-threshold", type=float, default=0.85)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--include-original-best",
        action="store_true",
        help="Also evaluate the original full-model best-budget circuit on BLiMP named entities.",
    )
    parser.add_argument(
        "--include-named-entity-best",
        action="store_true",
        help="Also evaluate the named-entity best-budget circuit on BLiMP named entities.",
    )
    parser.add_argument(
        "--extra-circuit-label",
        default=None,
        help="Optional label for an additional circuit to evaluate.",
    )
    parser.add_argument(
        "--extra-circuit-edge-path",
        default=None,
        help="Ranked edge CSV for an additional circuit.",
    )
    parser.add_argument(
        "--extra-circuit-budget-path",
        default=None,
        help="Budget sweep CSV for an additional circuit when resolving threshold/best.",
    )
    parser.add_argument(
        "--extra-circuit-budget-mode",
        choices=("threshold", "best", "fixed"),
        default="fixed",
        help="How to select the additional circuit budget.",
    )
    parser.add_argument(
        "--extra-circuit-budget",
        type=int,
        default=None,
        help="Fixed collapsed-edge budget for the additional circuit.",
    )
    parser.add_argument("--output-day", default=None)
    return parser.parse_args()


def safe_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return slug or "circuit"


def best_budget_row(budget_frame: pd.DataFrame) -> dict[str, Any]:
    if budget_frame.empty:
        raise ValueError("Budget sweep is empty.")
    row = (
        budget_frame.sort_values(
            ["faithfulness_mean", "accuracy_mean", "induced_node_count", "collapsed_edge_budget"],
            ascending=[False, False, True, True],
        )
        .iloc[0]
        .to_dict()
    )
    row["collapsed_edge_budget"] = int(row["collapsed_edge_budget"])
    row["faithfulness_mean"] = float(row["faithfulness_mean"])
    row["accuracy_mean"] = float(row["accuracy_mean"])
    return row


def resolve_named_entity_summary_path(
    project_root: Path,
    model_name: str,
    dataset_set: str,
    summary_path: str | None,
) -> Path:
    if summary_path is not None:
        return Path(summary_path)

    model_slug = safe_model_name(model_name)
    base = project_root / "results" / "named_entity_discovery" / model_slug / dataset_set
    candidates = sorted(
        base.glob("*/named_entity_discovery_summary_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No named-entity discovery summaries found under {base}")
    return candidates[0]


def resolve_repo_relative(project_root: Path, path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == project_root.name:
        return project_root.parent / path
    return project_root / path


def truncate_terminal_determiner(prefix: str) -> tuple[str, str]:
    stripped = prefix.strip()
    for suffix, label in ((" some", "some"), (" the", "the")):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)], label
    raise ValueError(f"Expected BLiMP prefix ending with ' some' or ' the': {prefix!r}")


def prepare_blimp_named_entity_rows(
    raw_df: pd.DataFrame,
    tokenizer,
    max_examples: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, row in raw_df.reset_index(drop=True).iterrows():
        if max_examples is not None and len(rows) >= max_examples:
            break

        original_prefix = str(row.get("one_prefix_prefix", "")).strip()
        if not original_prefix:
            failures.append(
                {
                    "row": int(idx),
                    "pairID": row.get("pairID"),
                    "UID": row.get("UID"),
                    "original_prefix": original_prefix,
                    "failure_reason": "empty_prefix",
                }
            )
            continue

        try:
            truncated_prefix, stripped_suffix = truncate_terminal_determiner(original_prefix)
        except ValueError as error:
            failures.append(
                {
                    "row": int(idx),
                    "pairID": row.get("pairID"),
                    "UID": row.get("UID"),
                    "original_prefix": original_prefix,
                    "failure_reason": str(error),
                }
            )
            continue

        rows.append(
            {
                "pairID": str(row.get("pairID")),
                "UID": row.get("UID"),
                "field": row.get("field"),
                "linguistics_term": row.get("linguistics_term"),
                "sentence_good": row.get("sentence_good"),
                "sentence_bad": row.get("sentence_bad"),
                "original_prefix": original_prefix,
                "prefix": truncated_prefix,
                "stripped_suffix": stripped_suffix,
                "seq_len": int(token_count_no_special(tokenizer, truncated_prefix)),
            }
        )

    prepared = pd.DataFrame(rows)
    if not prepared.empty:
        prepared = prepared.sort_values(["seq_len", "pairID"]).reset_index(drop=True)
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "row",
            "pairID",
            "UID",
            "original_prefix",
            "failure_reason",
        ],
    )
    return prepared, failure_frame


def build_circuit_specs(
    *,
    project_root: Path,
    model_name: str,
    dataset_set: str,
    original_main_experiment_path: str | None,
    named_entity_summary_path: str | None,
    source_faithfulness_threshold: float,
    include_original_best: bool,
    include_named_entity_best: bool,
    extra_circuit_label: str | None,
    extra_circuit_edge_path: str | None,
    extra_circuit_budget_path: str | None,
    extra_circuit_budget_mode: str,
    extra_circuit_budget: int | None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    original_paths = resolve_valid_source_artifacts(
        project_root,
        model_name,
        dataset_set,
        original_main_experiment_path,
    )
    original_budget_frame = pd.read_csv(original_paths["budget_path"])
    original_first = first_budget_reaching_faithfulness(
        original_budget_frame,
        source_faithfulness_threshold,
    )
    specs.append(
        {
            "label": "original_85_transfer",
            "display_label": "Original 85% transfer",
            "source_kind": "original_animacy",
            "edge_path": Path(original_paths["edge_path"]),
            "budget": int(original_first["collapsed_edge_budget"]),
            "budget_row": dict(original_first),
            "source_path": str(original_paths["source_dir"]),
        }
    )
    if include_original_best:
        original_best = best_budget_row(original_budget_frame)
        specs.append(
            {
                "label": "original_best_transfer",
                "display_label": "Original best transfer",
                "source_kind": "original_animacy",
                "edge_path": Path(original_paths["edge_path"]),
                "budget": int(original_best["collapsed_edge_budget"]),
                "budget_row": dict(original_best),
                "source_path": str(original_paths["source_dir"]),
            }
        )

    named_summary_path = resolve_named_entity_summary_path(
        project_root,
        model_name,
        dataset_set,
        named_entity_summary_path,
    )
    named_summary = json.loads(named_summary_path.read_text(encoding="utf-8"))
    named_source = named_summary["overlap"]["named_entity_source"]
    named_edge_path = resolve_repo_relative(project_root, named_source["edge_path"])
    named_first = named_summary["budget_sweep"]["first_threshold_row"]
    named_best = named_summary["budget_sweep"]["best_budget_row"]

    if named_first is None:
        raise ValueError(f"Named-entity discovery summary does not contain a first-threshold row: {named_summary_path}")

    specs.append(
        {
            "label": "named_entity_85",
            "display_label": "Named-entity 85%",
            "source_kind": "named_entity",
            "edge_path": named_edge_path,
            "budget": int(named_first["collapsed_edge_budget"]),
            "budget_row": dict(named_first),
            "source_path": str(named_summary_path),
        }
    )
    if include_named_entity_best:
        specs.append(
            {
                "label": "named_entity_best",
                "display_label": "Named-entity best",
                "source_kind": "named_entity",
                "edge_path": named_edge_path,
                "budget": int(named_best["collapsed_edge_budget"]),
                "budget_row": dict(named_best),
                "source_path": str(named_summary_path),
            }
        )

    if extra_circuit_edge_path is not None:
        if extra_circuit_label is None:
            raise ValueError("--extra-circuit-label is required when --extra-circuit-edge-path is set.")
        edge_path = resolve_repo_relative(project_root, extra_circuit_edge_path)
        budget_row: dict[str, Any] | None = None
        if extra_circuit_budget_mode == "fixed":
            if extra_circuit_budget is None:
                raise ValueError("--extra-circuit-budget is required when --extra-circuit-budget-mode=fixed.")
            budget = int(extra_circuit_budget)
        else:
            if extra_circuit_budget_path is None:
                raise ValueError("--extra-circuit-budget-path is required for threshold/best extra budgets.")
            budget_frame = pd.read_csv(resolve_repo_relative(project_root, extra_circuit_budget_path))
            if extra_circuit_budget_mode == "threshold":
                budget_row = first_budget_reaching_faithfulness(
                    budget_frame,
                    source_faithfulness_threshold,
                )
            else:
                budget_row = best_budget_row(budget_frame)
            budget = int(budget_row["collapsed_edge_budget"])
        specs.append(
            {
                "label": safe_label(extra_circuit_label),
                "display_label": extra_circuit_label,
                "source_kind": "extra",
                "edge_path": edge_path,
                "budget": budget,
                "budget_row": budget_row,
                "source_path": str(edge_path),
            }
        )

    return specs


def main() -> None:
    args = parse_args()
    project_root = resolve_animacy_circuit_root(Path.cwd())
    model_name = canonical_model_name(args.model)
    model_slug = safe_model_name(model_name)
    target_slug = target_source_slug(project_root, args.target_source)
    day = args.output_day or f"blimp_named_entity_transfer_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}"
    output_dir = (
        project_root
        / "results"
        / "blimp_named_entity_transfer"
        / model_slug
        / args.dataset_set
        / day
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_name)
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError(f"Model {model_name} has no tokenizer attached.")

    animate_ids, inanimate_ids, target_summary, target_path = named_entity_target_tensors(
        project_root,
        args.target_source,
        tokenizer,
        model.cfg.device,
        args.target_token_mode,
    )

    raw_df, dataset_path = load_local_blimp_prefix_dataset(project_root, args.blimp_config)
    prepared_rows, prefix_failures = prepare_blimp_named_entity_rows(
        raw_df,
        tokenizer,
        args.max_examples,
    )
    if prepared_rows.empty:
        raise ValueError("No valid BLiMP prefixes remained after truncating the terminal determiner.")

    circuits = build_circuit_specs(
        project_root=project_root,
        model_name=model_name,
        dataset_set=args.dataset_set,
        original_main_experiment_path=args.original_main_experiment_path,
        named_entity_summary_path=args.named_entity_summary_path,
        source_faithfulness_threshold=args.source_faithfulness_threshold,
        include_original_best=args.include_original_best,
        include_named_entity_best=args.include_named_entity_best,
        extra_circuit_label=args.extra_circuit_label,
        extra_circuit_edge_path=args.extra_circuit_edge_path,
        extra_circuit_budget_path=args.extra_circuit_budget_path,
        extra_circuit_budget_mode=args.extra_circuit_budget_mode,
        extra_circuit_budget=args.extra_circuit_budget,
    )

    prepared_rows_path = output_dir / f"blimp_named_entity_rows_{day}.csv"
    failures_path = output_dir / f"blimp_named_entity_prefix_failures_{day}.csv"
    summary_csv_path = output_dir / f"blimp_named_entity_transfer_summary_{day}.csv"
    summary_json_path = output_dir / f"blimp_named_entity_transfer_summary_{day}.json"
    status_path = output_dir / f"blimp_named_entity_transfer_status_{day}.json"
    save_csv(prepared_rows, prepared_rows_path, index=False)
    save_csv(prefix_failures, failures_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    circuit_status_rows: list[dict[str, Any]] = []
    for spec in circuits:
        edge_frame = pd.read_csv(spec["edge_path"])
        ranked_edges = parse_ranked_edge_frame(edge_frame)
        circuit_graph = build_budget_circuit(
            build_graph(model),
            ranked_edges,
            int(spec["budget"]),
        )
        result_rows = evaluate_blimp_passive_prefix_control(
            model=model,
            graph=circuit_graph,
            df=prepared_rows,
            animate_ids_tensor=animate_ids,
            inanimate_ids_tensor=inanimate_ids,
            batch_size=int(args.evaluation_batch_size),
        )
        circuit_summary = summarize_blimp_passive_prefix_control(result_rows)
        rows_path = output_dir / f"{safe_label(spec['label'])}_rows_{day}.csv"
        save_csv(result_rows, rows_path, index=False)

        budget_row = spec.get("budget_row") or {}
        summary_rows.append(
            {
                "model": model_name,
                "circuit_label": spec["label"],
                "display_label": spec["display_label"],
                "source_kind": spec["source_kind"],
                "source_path": spec["source_path"],
                "collapsed_edge_budget": int(spec["budget"]),
                "source_budget_fraction": budget_row.get("budget_fraction"),
                "source_faithfulness_mean": budget_row.get("faithfulness_mean"),
                "source_accuracy_mean": budget_row.get("accuracy_mean"),
                "source_validation_examples": budget_row.get("validation_examples"),
                **circuit_summary,
                "rows_path": str(rows_path),
            }
        )
        circuit_status_rows.append(
            {
                "label": spec["label"],
                "display_label": spec["display_label"],
                "source_kind": spec["source_kind"],
                "source_path": spec["source_path"],
                "edge_path": str(spec["edge_path"]),
                "budget": int(spec["budget"]),
                "budget_row": budget_row,
                "rows_path": str(rows_path),
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    save_csv(summary_frame, summary_csv_path, index=False)

    status = {
        "experiment": "blimp_named_entity_transfer",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset_set": args.dataset_set,
        "blimp_config": args.blimp_config,
        "target_source": str(args.target_source),
        "target_source_slug": target_slug,
        "target_source_path": str(target_path),
        "target_token_mode": args.target_token_mode,
        "target_counts": {
            "animate": int(animate_ids.numel()),
            "inanimate": int(inanimate_ids.numel()),
        },
        "target_filter_summary": target_summary,
        "dataset_counts": {
            "raw_rows": int(len(raw_df)),
            "valid_rows": int(len(prepared_rows)),
            "filtered_prefix_rows": int(len(prefix_failures)),
            "unique_original_prefixes": int(prepared_rows["original_prefix"].nunique()),
            "unique_truncated_prefixes": int(prepared_rows["prefix"].nunique()),
        },
        "circuits": circuit_status_rows,
        "paths": {
            "output_dir": str(output_dir),
            "prepared_rows": str(prepared_rows_path),
            "prefix_failures": str(failures_path),
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "status": str(status_path),
            "dataset_source": str(dataset_path),
        },
    }
    save_json(summary_json_path, {"rows": summary_rows})
    save_json(status_path, status)


if __name__ == "__main__":
    main()

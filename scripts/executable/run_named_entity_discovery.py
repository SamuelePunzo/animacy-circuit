from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circuit_finder_core import (  # noqa: E402
    DEFAULT_EAP_BUDGET_FLOOR,
    DEFAULT_EAP_BUDGET_MAX_FRACTION,
    DEFAULT_EAP_BUDGET_TAIL_POINTS,
    MODEL_SPECIFIC_CORRECT,
    attribute_graph,
    build_graph,
    canonical_model_name,
    collapsed_edge_groups,
    compute_sequence_metrics,
    first_budget_reaching_faithfulness,
    induced_node_ranking,
    load_model,
    make_dataloader,
    make_eap_metrics,
    parse_ranked_edge_frame,
    ranking_frame,
    resolve_animacy_circuit_root,
    resolve_eap_budget_grid,
    safe_model_name,
    sample_discovery_validation,
    save_csv,
    save_eap_visualizations,
    save_json,
)
from evaluate_named_entity_circuit import (  # noqa: E402
    filter_model_success,
    load_truncated_pairs,
    named_entity_target_tensors,
    resolve_valid_source_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-model EAP-IG discovery on the named-entity truncated-prefix task "
            "and compare the resulting circuit with the original target-set circuit."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset-set", default=MODEL_SPECIFIC_CORRECT)
    parser.add_argument(
        "--target-source",
        default="dataset/semantic_meaningful/named_entity_targets.json",
    )
    parser.add_argument(
        "--original-main-experiment-path",
        default="animacy-circuit/results/eap_ig/gpt2/model_specific_correct/2026-05-30/full_model",
        help="Original/common-noun full_model EAP run to compare against.",
    )
    parser.add_argument("--source-faithfulness-threshold", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--attribution-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--discovery-sample-size", type=int, default=500)
    parser.add_argument("--discovery-margin-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--target-token-mode",
        choices=("first_token", "whole_entity_single_token"),
        default="first_token",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--budget-max-fraction", type=float, default=DEFAULT_EAP_BUDGET_MAX_FRACTION)
    parser.add_argument("--budget-floor", type=int, default=DEFAULT_EAP_BUDGET_FLOOR)
    parser.add_argument("--budget-tail-points", type=int, default=DEFAULT_EAP_BUDGET_TAIL_POINTS)
    parser.add_argument("--output-day", default=None)
    return parser.parse_args()


def edge_names(edges: Sequence[dict[str, Any]], budget: int | None = None) -> list[str]:
    selected = edges if budget is None else edges[:budget]
    return [str(edge["collapsed_edge"]) for edge in selected]


def edge_set_overlap(
    left_edges: Sequence[dict[str, Any]],
    right_edges: Sequence[dict[str, Any]],
    *,
    left_budget: int,
    right_budget: int,
) -> dict[str, Any]:
    left = set(edge_names(left_edges, left_budget))
    right = set(edge_names(right_edges, right_budget))
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


def rank_overlap_summary(
    original_edges: Sequence[dict[str, Any]],
    named_edges: Sequence[dict[str, Any]],
    budgets: Sequence[int],
) -> dict[str, Any]:
    rows = {}
    for budget in budgets:
        if budget <= len(original_edges) and budget <= len(named_edges):
            rows[f"top_{budget}"] = edge_set_overlap(
                original_edges,
                named_edges,
                left_budget=budget,
                right_budget=budget,
            )
    return rows


def best_budget_row(budget_frame: pd.DataFrame) -> dict[str, Any]:
    if budget_frame.empty:
        return {}
    row = budget_frame.sort_values("faithfulness_mean", ascending=False).iloc[0].to_dict()
    row["collapsed_edge_budget"] = int(row["collapsed_edge_budget"])
    row["faithfulness_mean"] = float(row["faithfulness_mean"])
    return row


def main() -> None:
    args = parse_args()
    project_root = resolve_animacy_circuit_root(Path.cwd())
    model_name = canonical_model_name(args.model)
    model_slug = safe_model_name(model_name)
    day = args.output_day or f"named_entity_discovery_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}"
    output_dir = (
        project_root
        / "results"
        / "named_entity_discovery"
        / model_slug
        / args.dataset_set
        / day
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    original_paths = resolve_valid_source_artifacts(
        project_root,
        model_name,
        args.dataset_set,
        args.original_main_experiment_path,
    )
    original_edge_frame = pd.read_csv(original_paths["edge_path"])
    original_budget_frame = pd.read_csv(original_paths["budget_path"])
    original_edges = parse_ranked_edge_frame(original_edge_frame)
    original_first_threshold = first_budget_reaching_faithfulness(
        original_budget_frame,
        args.source_faithfulness_threshold,
    )
    original_85_budget = int(original_first_threshold["collapsed_edge_budget"])

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

    raw_df = load_truncated_pairs(project_root, model_name, args.max_examples)
    from circuit_finder_core import add_sequence_lengths

    raw_df = add_sequence_lengths(raw_df, model)
    scored_df = compute_sequence_metrics(
        raw_df,
        model,
        tokenizer,
        animate_ids,
        inanimate_ids,
        batch_size=args.batch_size,
    )
    retained_df = filter_model_success(scored_df)
    discovery_df, validation_df, sample_signature = sample_discovery_validation(
        retained_df,
        discovery_sample_size=args.discovery_sample_size,
        seed=args.seed,
        discovery_margin_threshold=args.discovery_margin_threshold,
    )

    scored_path = output_dir / f"named_entity_truncated_scored_{day}.csv"
    retained_path = output_dir / f"named_entity_model_success_{day}.csv"
    discovery_path = output_dir / f"named_entity_discovery_sample_{day}.csv"
    validation_path = output_dir / f"named_entity_validation_{day}.csv"
    save_csv(scored_df, scored_path, index=False)
    save_csv(retained_df, retained_path, index=False)
    save_csv(discovery_df, discovery_path, index=False)
    save_csv(validation_df, validation_path, index=False)

    metrics = make_eap_metrics(animate_ids, inanimate_ids)
    discovery_loader = make_dataloader(
        discovery_df,
        batch_size=args.attribution_batch_size,
        shuffle=False,
    )
    validation_loader = make_dataloader(
        validation_df,
        batch_size=args.evaluation_batch_size,
        shuffle=False,
    )

    graph = attribute_graph(
        model=model,
        graph=build_graph(model),
        dataloader=discovery_loader,
        metric=metrics["attribute"],
        ig_steps=args.ig_steps,
    )
    named_edges = collapsed_edge_groups(graph)
    named_nodes = induced_node_ranking(named_edges)
    edge_frame = ranking_frame(named_edges)
    node_frame = ranking_frame(named_nodes)

    edge_path = output_dir / f"named_entity_full_model_edges_{day}.csv"
    node_path = output_dir / f"named_entity_full_model_nodes_{day}.csv"
    budget_path = output_dir / f"named_entity_budget_sweep_{day}.csv"
    summary_path = output_dir / f"named_entity_discovery_summary_{day}.json"
    overlap_path = output_dir / f"named_entity_original_overlap_{day}.json"
    save_csv(edge_frame, edge_path, index=False)
    save_csv(node_frame, node_path, index=False)

    from circuit_finder_core import run_eap_budget_sweep

    budget_grid = resolve_eap_budget_grid(
        len(named_edges),
        budgets=args.budgets,
        budget_max_fraction=args.budget_max_fraction,
        budget_floor=args.budget_floor,
        budget_tail_points=args.budget_tail_points,
    )
    budget_frame, early_stop_summary = run_eap_budget_sweep(
        model=model,
        scored_graph=graph,
        ranked_edges=named_edges,
        validation_loader=validation_loader,
        faithfulness_metric=metrics["faithfulness"],
        accuracy_metric=metrics["accuracy"],
        budgets=budget_grid,
        early_stop=False,
    )
    save_csv(budget_frame, budget_path, index=False)

    named_first_threshold: dict[str, Any] | None
    try:
        named_first_threshold = first_budget_reaching_faithfulness(
            budget_frame,
            args.source_faithfulness_threshold,
        )
    except ValueError:
        named_first_threshold = None
    named_reference_budget = (
        int(named_first_threshold["collapsed_edge_budget"])
        if named_first_threshold is not None
        else int(best_budget_row(budget_frame).get("collapsed_edge_budget", min(original_85_budget, len(named_edges))))
    )

    overlap = {
        "original_source": {
            "edge_path": str(original_paths["edge_path"]),
            "budget_path": str(original_paths["budget_path"]),
            "first_threshold_row": original_first_threshold,
            "circuit_budget": original_85_budget,
        },
        "named_entity_source": {
            "edge_path": str(edge_path),
            "budget_path": str(budget_path),
            "first_threshold_row": named_first_threshold,
            "best_budget_row": best_budget_row(budget_frame),
            "reference_budget": named_reference_budget,
        },
        "same_rank_overlap": rank_overlap_summary(
            original_edges,
            named_edges,
            budgets=(30, 50, 100, 200, 500, 1000, original_85_budget),
        ),
        "circuit_overlap": {
            "original_85_vs_named_reference": edge_set_overlap(
                original_edges,
                named_edges,
                left_budget=original_85_budget,
                right_budget=named_reference_budget,
            ),
            "same_budget_original_85": edge_set_overlap(
                original_edges,
                named_edges,
                left_budget=original_85_budget,
                right_budget=min(original_85_budget, len(named_edges)),
            ),
        },
    }
    save_json(overlap_path, overlap)

    visualization_paths = save_eap_visualizations(
        project_root=project_root,
        output_dir=output_dir,
        edge_frame=edge_frame,
        node_frame=node_frame,
        budget_frame=budget_frame,
        day=day,
    )

    summary = {
        "experiment": "named_entity_full_model_discovery",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset_set": args.dataset_set,
        "target_source": str(args.target_source),
        "target_source_path": str(target_path),
        "target_token_mode": args.target_token_mode,
        "target_counts": {
            "animate": int(animate_ids.numel()),
            "inanimate": int(inanimate_ids.numel()),
        },
        "target_summary": target_summary,
        "dataset_counts": {
            "truncated_scored": int(len(scored_df)),
            "model_success": int(len(retained_df)),
            "discovery": int(len(discovery_df)),
            "validation": int(len(validation_df)),
            "discovery_sample_signature": sample_signature,
            "discovery_margin_threshold": args.discovery_margin_threshold,
        },
        "config": vars(args),
        "budget_sweep": {
            "budget_count": int(len(budget_frame)),
            "first_threshold_row": named_first_threshold,
            "best_budget_row": best_budget_row(budget_frame),
            "early_stop": early_stop_summary,
        },
        "overlap": overlap,
        "paths": {
            "output_dir": str(output_dir),
            "scored_dataset": str(scored_path),
            "retained_dataset": str(retained_path),
            "discovery_dataset": str(discovery_path),
            "validation_dataset": str(validation_path),
            "edge_rankings": str(edge_path),
            "node_rankings": str(node_path),
            "budget_sweep": str(budget_path),
            "overlap": str(overlap_path),
            "summary": str(summary_path),
            "visualizations": visualization_paths,
        },
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

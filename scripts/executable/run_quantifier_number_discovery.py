from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circuit_finder_core import (  # noqa: E402
    DEFAULT_EAP_BUDGET_FLOOR,
    DEFAULT_EAP_BUDGET_MAX_FRACTION,
    DEFAULT_EAP_BUDGET_TAIL_POINTS,
    MODEL_SPECIFIC_CORRECT,
    add_sequence_lengths,
    attribute_graph,
    build_graph,
    canonical_model_name,
    collapsed_edge_groups,
    compute_sequence_metrics,
    first_budget_reaching_faithfulness,
    generate_exact_length_batches,
    induced_node_ranking,
    load_model,
    make_dataloader,
    make_eap_metrics,
    parse_ranked_edge_frame,
    ranking_frame,
    resolve_animacy_circuit_root,
    resolve_eap_budget_grid,
    resolve_target_source_path,
    safe_model_name,
    sample_discovery_validation,
    save_csv,
    save_eap_visualizations,
    save_json,
    tokenization_filter_jsonl_pairs_path,
)
from evaluate_named_entity_circuit import (  # noqa: E402
    filter_model_success,
    resolve_valid_source_artifacts,
)
from run_named_entity_discovery import (  # noqa: E402
    best_budget_row,
    edge_set_overlap,
    rank_overlap_summary,
)


TARGET_CLASSES = ("animate", "inanimate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EAP-IG discovery on the by-many/by-a quantifier number-control task "
            "for either animate or inanimate singular/plural target pairs."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset-set", default=MODEL_SPECIFIC_CORRECT)
    parser.add_argument(
        "--target-source",
        default="dataset/semantic_meaningful/quantifier_number_targets.json",
        help="Quantifier target-pair JSON path.",
    )
    parser.add_argument("--target-class", choices=TARGET_CLASSES, required=True)
    parser.add_argument(
        "--original-main-experiment-path",
        default="animacy-circuit/results/eap_ig/gpt2/model_specific_correct/2026-05-30/full_model",
        help="Original/common-noun full_model EAP run to compare against.",
    )
    parser.add_argument("--source-faithfulness-threshold", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-score-batch-size", type=int, default=None)
    parser.add_argument("--attribution-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--discovery-sample-size", type=int, default=500)
    parser.add_argument("--discovery-margin-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--budget-max-fraction", type=float, default=DEFAULT_EAP_BUDGET_MAX_FRACTION)
    parser.add_argument("--budget-floor", type=int, default=DEFAULT_EAP_BUDGET_FLOOR)
    parser.add_argument("--budget-tail-points", type=int, default=DEFAULT_EAP_BUDGET_TAIL_POINTS)
    parser.add_argument("--output-day", default=None)
    return parser.parse_args()


def strip_terminal_by_the(prefix: str) -> str:
    suffix = " by the"
    if not prefix.endswith(suffix):
        raise ValueError(f"Expected prefix to end with {suffix!r}: {prefix!r}")
    return prefix[: -len(suffix)]


def load_quantifier_pairs_dataframe(
    project_root: Path,
    model_name: str,
    max_examples: int | None,
) -> pd.DataFrame:
    path = tokenization_filter_jsonl_pairs_path(project_root, model_name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing tokenization-filtered pair file: {path}")

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            clean = item.get("clean") or item.get("clean_prefix")
            corrupt = item.get("corrupt") or item.get("corrupt_prefix")
            if clean is None or corrupt is None:
                continue
            row = dict(item)
            row["original_clean_prefix"] = clean
            row["original_corrupt_prefix"] = corrupt
            row["clean_prefix"] = f"{strip_terminal_by_the(str(clean))} by many"
            row["corrupt_prefix"] = f"{strip_terminal_by_the(str(corrupt))} by a"
            rows.append(row)
            if max_examples is not None and len(rows) >= max_examples:
                break

    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return pd.DataFrame(rows).drop_duplicates(subset=["clean_prefix", "corrupt_prefix"]).reset_index(drop=True)


def load_raw_quantifier_targets(project_root: Path, target_source: str) -> tuple[dict[str, list[dict[str, str]]], Path]:
    path = resolve_target_source_path(project_root, target_source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise ValueError(f"Quantifier target source has no object-valued 'targets': {path}")
    missing = [target_class for target_class in TARGET_CLASSES if target_class not in targets]
    if missing:
        raise ValueError(f"Quantifier target source is missing classes {missing}: {path}")
    parsed: dict[str, list[dict[str, str]]] = {}
    for target_class in TARGET_CLASSES:
        parsed[target_class] = [
            {"singular": str(row["singular"]), "plural": str(row["plural"])}
            for row in targets[target_class]
        ]
    return parsed, path


def one_token_id(tokenizer, text: str) -> tuple[int | None, list[int]]:
    token_ids = tokenizer(" " + text, add_special_tokens=False).input_ids
    token_ids = [int(token_id) for token_id in token_ids]
    if len(token_ids) != 1:
        return None, token_ids
    return token_ids[0], token_ids


def tokenizer_filter_target_pairs(
    pairs: Sequence[dict[str, str]],
    tokenizer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_singular_ids: set[int] = set()
    seen_plural_ids: set[int] = set()
    for row in pairs:
        singular = row["singular"]
        plural = row["plural"]
        singular_id, singular_tokens = one_token_id(tokenizer, singular)
        plural_id, plural_tokens = one_token_id(tokenizer, plural)
        if singular_id is None:
            dropped.append(
                {
                    "singular": singular,
                    "plural": plural,
                    "reason": "singular_not_one_token",
                    "singular_token_ids": singular_tokens,
                    "plural_token_ids": plural_tokens,
                }
            )
            continue
        if plural_id is None:
            dropped.append(
                {
                    "singular": singular,
                    "plural": plural,
                    "reason": "plural_not_one_token",
                    "singular_token_ids": singular_tokens,
                    "plural_token_ids": plural_tokens,
                }
            )
            continue
        if singular_id in seen_singular_ids:
            dropped.append(
                {
                    "singular": singular,
                    "plural": plural,
                    "reason": "duplicate_singular_token_id",
                    "singular_token_id": singular_id,
                    "plural_token_id": plural_id,
                }
            )
            continue
        if plural_id in seen_plural_ids:
            dropped.append(
                {
                    "singular": singular,
                    "plural": plural,
                    "reason": "duplicate_plural_token_id",
                    "singular_token_id": singular_id,
                    "plural_token_id": plural_id,
                }
            )
            continue
        seen_singular_ids.add(singular_id)
        seen_plural_ids.add(plural_id)
        kept.append(
            {
                "singular": singular,
                "plural": plural,
                "singular_token_id": singular_id,
                "plural_token_id": plural_id,
                "singular_token": tokenizer.decode([singular_id]).strip(),
                "plural_token": tokenizer.decode([plural_id]).strip(),
            }
        )
    return kept, dropped


def score_target_pairs(
    df: pd.DataFrame,
    model,
    pairs: Sequence[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    if not pairs:
        return []
    singular_ids = torch.tensor(
        [int(row["singular_token_id"]) for row in pairs],
        dtype=torch.long,
        device=model.cfg.device,
    )
    plural_ids = torch.tensor(
        [int(row["plural_token_id"]) for row in pairs],
        dtype=torch.long,
        device=model.cfg.device,
    )
    singular_sum = torch.zeros(len(pairs), dtype=torch.float64, device=model.cfg.device)
    plural_sum = torch.zeros(len(pairs), dtype=torch.float64, device=model.cfg.device)
    example_count = 0

    estimated_batches = sum(max(1, (len(group) + batch_size - 1) // batch_size) for _, group in df.groupby("seq_len"))
    for clean_tokens, corrupt_tokens, batch_df in tqdm(
        generate_exact_length_batches(
            df=df,
            model=model,
            batch_size=batch_size,
            device=model.cfg.device,
        ),
        total=estimated_batches,
        desc="Scoring target-pair compatibility",
    ):
        with torch.no_grad():
            clean_logits = model(clean_tokens)[:, -1, :]
            corrupt_logits = model(corrupt_tokens)[:, -1, :]
        plural_sum += clean_logits[:, plural_ids].double().sum(dim=0)
        singular_sum += corrupt_logits[:, singular_ids].double().sum(dim=0)
        example_count += int(len(batch_df))

    if example_count == 0:
        raise ValueError("Cannot score target pairs with zero prompt examples.")

    scored: list[dict[str, Any]] = []
    for idx, row in enumerate(pairs):
        singular_mean = float((singular_sum[idx] / example_count).item())
        plural_mean = float((plural_sum[idx] / example_count).item())
        scored.append(
            {
                **row,
                "singular_context_mean_logit": singular_mean,
                "plural_context_mean_logit": plural_mean,
                "pair_score": singular_mean + plural_mean,
            }
        )
    return scored


def select_matched_targets(
    scored_by_class: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    matched_count = min(len(scored_by_class[target_class]) for target_class in TARGET_CLASSES)
    if matched_count <= 0:
        raise ValueError("No matched quantifier target pairs are available after tokenizer filtering.")
    selected: dict[str, list[dict[str, Any]]] = {}
    for target_class in TARGET_CLASSES:
        selected[target_class] = sorted(
            scored_by_class[target_class],
            key=lambda row: (-float(row["pair_score"]), row["singular"], row["plural"]),
        )[:matched_count]
    return selected, {
        "matched_count": int(matched_count),
        "policy": "keep_highest_pair_score_per_class",
        "pair_score": "mean_logit(singular | by a) + mean_logit(plural | by many)",
        "pre_match_counts": {
            target_class: int(len(scored_by_class[target_class]))
            for target_class in TARGET_CLASSES
        },
        "discarded_counts": {
            target_class: int(len(scored_by_class[target_class]) - matched_count)
            for target_class in TARGET_CLASSES
        },
        "lowest_kept_scores": {
            target_class: (
                float(selected[target_class][-1]["pair_score"])
                if selected[target_class]
                else None
            )
            for target_class in TARGET_CLASSES
        },
    }


def target_tensors(
    selected_pairs: Sequence[dict[str, Any]],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    plural_ids = torch.tensor(
        [int(row["plural_token_id"]) for row in selected_pairs],
        dtype=torch.long,
        device=device,
    )
    singular_ids = torch.tensor(
        [int(row["singular_token_id"]) for row in selected_pairs],
        dtype=torch.long,
        device=device,
    )
    if plural_ids.numel() == 0 or singular_ids.numel() == 0:
        raise ValueError("Selected target tensors are empty.")
    return plural_ids, singular_ids


def target_selection_summary(
    raw_targets: dict[str, list[dict[str, str]]],
    tokenizer_kept: dict[str, list[dict[str, Any]]],
    tokenizer_dropped: dict[str, list[dict[str, Any]]],
    scored_by_class: dict[str, list[dict[str, Any]]],
    selected_by_class: dict[str, list[dict[str, Any]]],
    match_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "matching": match_summary,
        "classes": {
            target_class: {
                "raw_count": int(len(raw_targets[target_class])),
                "tokenizer_kept_count": int(len(tokenizer_kept[target_class])),
                "tokenizer_dropped_count": int(len(tokenizer_dropped[target_class])),
                "scored_count": int(len(scored_by_class[target_class])),
                "selected_count": int(len(selected_by_class[target_class])),
                "kept_examples": selected_by_class[target_class][:20],
                "dropped_examples": tokenizer_dropped[target_class][:20],
            }
            for target_class in TARGET_CLASSES
        },
    }


def main() -> None:
    args = parse_args()
    project_root = resolve_animacy_circuit_root(Path.cwd())
    model_name = canonical_model_name(args.model)
    model_slug = safe_model_name(model_name)
    day = args.output_day or f"quantifier_number_{args.target_class}_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}"
    output_dir = (
        project_root
        / "results"
        / "quantifier_number_discovery"
        / model_slug
        / args.dataset_set
        / args.target_class
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

    raw_targets, target_path = load_raw_quantifier_targets(project_root, args.target_source)
    raw_df = load_quantifier_pairs_dataframe(project_root, model_name, args.max_examples)
    raw_df = add_sequence_lengths(raw_df, model)
    if raw_df.empty:
        raise ValueError("No length-aligned quantifier prompt pairs remain after tokenization.")

    tokenizer_kept: dict[str, list[dict[str, Any]]] = {}
    tokenizer_dropped: dict[str, list[dict[str, Any]]] = {}
    scored_by_class: dict[str, list[dict[str, Any]]] = {}
    target_score_batch_size = args.target_score_batch_size or args.batch_size
    for target_class in TARGET_CLASSES:
        kept, dropped = tokenizer_filter_target_pairs(raw_targets[target_class], tokenizer)
        tokenizer_kept[target_class] = kept
        tokenizer_dropped[target_class] = dropped
        scored_by_class[target_class] = score_target_pairs(
            raw_df,
            model,
            kept,
            batch_size=target_score_batch_size,
        )

    selected_by_class, match_summary = select_matched_targets(scored_by_class)
    selected_pairs = selected_by_class[args.target_class]
    plural_ids, singular_ids = target_tensors(selected_pairs, model.cfg.device)

    scored_df = compute_sequence_metrics(
        raw_df,
        model,
        tokenizer,
        plural_ids,
        singular_ids,
        batch_size=args.batch_size,
    )
    retained_df = filter_model_success(scored_df)
    discovery_df, validation_df, sample_signature = sample_discovery_validation(
        retained_df,
        discovery_sample_size=args.discovery_sample_size,
        seed=args.seed,
        discovery_margin_threshold=args.discovery_margin_threshold,
    )

    prefix = f"quantifier_number_{args.target_class}"
    scored_path = output_dir / f"{prefix}_scored_{day}.csv"
    retained_path = output_dir / f"{prefix}_model_success_{day}.csv"
    discovery_path = output_dir / f"{prefix}_discovery_sample_{day}.csv"
    validation_path = output_dir / f"{prefix}_validation_{day}.csv"
    targets_path = output_dir / f"{prefix}_selected_targets_{day}.json"
    save_csv(scored_df, scored_path, index=False)
    save_csv(retained_df, retained_path, index=False)
    save_csv(discovery_df, discovery_path, index=False)
    save_csv(validation_df, validation_path, index=False)

    selection_summary = target_selection_summary(
        raw_targets,
        tokenizer_kept,
        tokenizer_dropped,
        scored_by_class,
        selected_by_class,
        match_summary,
    )
    save_json(targets_path, selection_summary)

    metrics = make_eap_metrics(plural_ids, singular_ids)
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
    quantifier_edges = collapsed_edge_groups(graph)
    quantifier_nodes = induced_node_ranking(quantifier_edges)
    edge_frame = ranking_frame(quantifier_edges)
    node_frame = ranking_frame(quantifier_nodes)

    edge_path = output_dir / f"{prefix}_full_model_edges_{day}.csv"
    node_path = output_dir / f"{prefix}_full_model_nodes_{day}.csv"
    budget_path = output_dir / f"{prefix}_budget_sweep_{day}.csv"
    summary_path = output_dir / f"{prefix}_discovery_summary_{day}.json"
    overlap_path = output_dir / f"{prefix}_original_overlap_{day}.json"
    save_csv(edge_frame, edge_path, index=False)
    save_csv(node_frame, node_path, index=False)

    from circuit_finder_core import run_eap_budget_sweep

    budget_grid = resolve_eap_budget_grid(
        len(quantifier_edges),
        budgets=args.budgets,
        budget_max_fraction=args.budget_max_fraction,
        budget_floor=args.budget_floor,
        budget_tail_points=args.budget_tail_points,
    )
    budget_frame, early_stop_summary = run_eap_budget_sweep(
        model=model,
        scored_graph=graph,
        ranked_edges=quantifier_edges,
        validation_loader=validation_loader,
        faithfulness_metric=metrics["faithfulness"],
        accuracy_metric=metrics["accuracy"],
        budgets=budget_grid,
        early_stop=False,
    )
    save_csv(budget_frame, budget_path, index=False)

    quantifier_first_threshold: dict[str, Any] | None
    try:
        quantifier_first_threshold = first_budget_reaching_faithfulness(
            budget_frame,
            args.source_faithfulness_threshold,
        )
    except ValueError:
        quantifier_first_threshold = None
    quantifier_reference_budget = (
        int(quantifier_first_threshold["collapsed_edge_budget"])
        if quantifier_first_threshold is not None
        else int(best_budget_row(budget_frame).get("collapsed_edge_budget", min(original_85_budget, len(quantifier_edges))))
    )

    overlap = {
        "original_source": {
            "edge_path": str(original_paths["edge_path"]),
            "budget_path": str(original_paths["budget_path"]),
            "first_threshold_row": original_first_threshold,
            "circuit_budget": original_85_budget,
        },
        "quantifier_number_source": {
            "target_class": args.target_class,
            "edge_path": str(edge_path),
            "budget_path": str(budget_path),
            "first_threshold_row": quantifier_first_threshold,
            "best_budget_row": best_budget_row(budget_frame),
            "reference_budget": quantifier_reference_budget,
        },
        "same_rank_overlap": rank_overlap_summary(
            original_edges,
            quantifier_edges,
            budgets=(30, 50, 100, 200, 500, 1000, original_85_budget),
        ),
        "circuit_overlap": {
            "original_85_vs_quantifier_reference": edge_set_overlap(
                original_edges,
                quantifier_edges,
                left_budget=original_85_budget,
                right_budget=quantifier_reference_budget,
            ),
            "same_budget_original_85": edge_set_overlap(
                original_edges,
                quantifier_edges,
                left_budget=original_85_budget,
                right_budget=min(original_85_budget, len(quantifier_edges)),
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
        "experiment": "quantifier_number_full_model_discovery",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset_set": args.dataset_set,
        "target_class": args.target_class,
        "target_source": str(args.target_source),
        "target_source_path": str(target_path),
        "target_counts": {
            "plural_clean_targets": int(plural_ids.numel()),
            "singular_corrupt_targets": int(singular_ids.numel()),
        },
        "target_selection": selection_summary,
        "dataset_counts": {
            "quantifier_scored": int(len(scored_df)),
            "model_success": int(len(retained_df)),
            "discovery": int(len(discovery_df)),
            "validation": int(len(validation_df)),
            "discovery_sample_signature": sample_signature,
            "discovery_margin_threshold": args.discovery_margin_threshold,
        },
        "metric": {
            "clean_prefix": "The [patient] was [clean_verb] by many",
            "corrupt_prefix": "The [patient] was [corrupt_verb] by a",
            "clean_targets": "plural nouns",
            "corrupt_targets": "singular nouns",
            "logit_difference": "mean(plural target logits) - mean(singular target logits)",
        },
        "config": vars(args),
        "budget_sweep": {
            "budget_count": int(len(budget_frame)),
            "first_threshold_row": quantifier_first_threshold,
            "best_budget_row": best_budget_row(budget_frame),
            "early_stop": early_stop_summary,
        },
        "overlap": overlap,
        "paths": {
            "output_dir": str(output_dir),
            "selected_targets": str(targets_path),
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

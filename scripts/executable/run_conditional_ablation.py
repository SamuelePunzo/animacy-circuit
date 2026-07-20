from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from circuit_finder_core import (
    DEFAULT_DISCOVERY_MARGIN_THRESHOLD,
    DEFAULT_TARGET_SOURCE,
    MODEL_SPECIFIC_CORRECT,
    build_graph,
    canonical_model_name,
    date_tag,
    ensure_dir,
    make_dataloader,
    make_eap_metrics,
    prepare_filtered_model_inputs,
    resolve_animacy_circuit_root,
    safe_model_name,
    sample_discovery_validation,
    save_csv,
    save_json,
    timestamp_tag,
    underlying_edge_name_set,
)
from run_eap_localization import (
    LocalizationConfig,
    _resolve_path,
    build_ablate_circuit,
    evaluate_circuit,
    load_ranked_edges_csv,
    make_kl_to_clean_metric,
    maybe_limit_validation,
    prepare_named_entity_truncated_inputs,
)


@dataclass
class ConditionalAblationConfig:
    model_name: str = "gpt2"
    dataset_filter_model_name: str = "gpt2"
    discovery_margin_threshold: float | None = DEFAULT_DISCOVERY_MARGIN_THRESHOLD
    filter_batch_size: int = 50
    evaluation_batch_size: int = 1
    max_validation_examples: int | None = None
    output_day: str | None = None
    dataset_filter_path: str | None = None
    refresh_dataset_filter: bool = False
    cache_dataset_filter: bool = True
    max_filter_examples: int | None = None
    target_filter_policy: str = "model_success"
    dataset_mode: str = "semantic_filtered"
    dataset_set: str = MODEL_SPECIFIC_CORRECT
    named_entity_discovery_dir: str | None = None
    target_source: str = DEFAULT_TARGET_SOURCE
    target_token_mode: str = "first_token"
    filtered_df_seed: int | None = None
    localization_source_path: str | None = None
    edge_rankings_path: str | None = None
    sample_size: int = 500
    seed: int = 42
    protected_budget: int = 20
    ablated_budget: int = 20
    candidate_start_rank: int = 21
    candidate_end_rank: int = 200
    band_size: int | None = None
    sample_count: int = 100
    sampling_strategy: str = "score_weighted"
    random_seed: int = 0


def conditional_ablation_root(project_root: Path, model_name: str, day: str | None = None) -> Path:
    resolved = canonical_model_name(model_name)
    return ensure_dir(
        project_root
        / "results"
        / "eap_ig_conditional_ablation"
        / safe_model_name(resolved)
        / (date_tag() if day is None else day)
    )


def edge_rank(edge_group: dict[str, Any]) -> int:
    return int(edge_group["rank"])


def edge_abs_score(edge_group: dict[str, Any]) -> float:
    return abs(float(edge_group["abs_score"]))


def edge_set_key(edge_groups: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(str(edge["collapsed_edge"]) for edge in edge_groups))


def resolve_localization_summary_path(
    project_root: Path,
    path_value: str | Path,
    sample_size: int,
    seed: int,
) -> Path:
    path = _resolve_path(project_root, path_value)
    if path.is_dir():
        candidate = path / f"sample_{sample_size}" / f"seed_{seed}" / (
            f"localization_summary_sample_{sample_size}_seed_{seed}.json"
        )
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"No localization summary found for sample_size={sample_size}, seed={seed} under {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Localization source path does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if {"sample_size", "seed", "paths"}.issubset(payload.keys()):
        summary_sample_size = int(payload["sample_size"])
        summary_seed = int(payload["seed"])
        if summary_sample_size != int(sample_size) or summary_seed != int(seed):
            raise ValueError(
                "Localization summary slot does not match requested sample/seed: "
                f"({summary_sample_size}, {summary_seed}) != ({sample_size}, {seed})"
            )
        return path

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Unsupported localization source: {path}")
    for run in runs:
        if int(run.get("sample_size", -1)) == int(sample_size) and int(run.get("seed", -1)) == int(seed):
            summary_value = run.get("summary")
            if not summary_value:
                raise ValueError(f"Localization manifest entry is missing a summary path for {sample_size}/{seed}")
            summary_path = _resolve_path(project_root, summary_value)
            if not summary_path.is_file():
                raise FileNotFoundError(f"Localization summary referenced by manifest does not exist: {summary_path}")
            return summary_path
    raise ValueError(f"No localization run found for sample_size={sample_size}, seed={seed} in {path}")


def filter_candidate_edges(
    ranked_edges: Sequence[dict[str, Any]],
    *,
    protected_budget: int,
    candidate_start_rank: int,
    candidate_end_rank: int,
) -> list[dict[str, Any]]:
    protected_edges = list(ranked_edges[:protected_budget])
    protected_collapsed = {str(edge["collapsed_edge"]) for edge in protected_edges}
    protected_underlying = underlying_edge_name_set(protected_edges)

    candidates: list[dict[str, Any]] = []
    for edge in ranked_edges:
        rank = edge_rank(edge)
        if rank < int(candidate_start_rank) or rank > int(candidate_end_rank):
            continue
        if str(edge["collapsed_edge"]) in protected_collapsed:
            continue
        edge_underlying = {str(name) for name in edge.get("underlying_edges", [])}
        if edge_underlying & protected_underlying:
            continue
        candidates.append(edge)
    return candidates


def build_rank_band_edge_sets(
    candidate_edges: Sequence[dict[str, Any]],
    *,
    set_size: int,
) -> list[dict[str, Any]]:
    ordered = sorted(candidate_edges, key=edge_rank)
    sets: list[dict[str, Any]] = []
    for start in range(0, len(ordered) - set_size + 1, set_size):
        edges = ordered[start : start + set_size]
        sets.append(
            {
                "set_id": f"rank_{edge_rank(edges[0])}_{edge_rank(edges[-1])}",
                "set_kind": "rank_band",
                "sampling_strategy": "deterministic",
                "edges": list(edges),
            }
        )
    return sets


def sample_edge_sets(
    candidate_edges: Sequence[dict[str, Any]],
    *,
    set_size: int,
    sample_count: int,
    strategy: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    if sample_count <= 0:
        return []
    pool = list(candidate_edges)
    if len(pool) < set_size:
        raise ValueError(f"Need at least {set_size} candidate edges, but only found {len(pool)}.")

    rng = np.random.default_rng(int(random_seed))
    weights = np.array([edge_abs_score(edge) for edge in pool], dtype=float)
    if strategy == "uniform" or not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        probabilities = None
    else:
        probabilities = weights / float(weights.sum())

    selected_sets: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    max_attempts = max(sample_count * 20, 100)
    attempts = 0
    while len(selected_sets) < sample_count and attempts < max_attempts:
        attempts += 1
        indices = rng.choice(len(pool), size=set_size, replace=False, p=probabilities)
        edges = sorted((pool[int(index)] for index in indices), key=edge_rank)
        key = edge_set_key(edges)
        if key in seen:
            continue
        seen.add(key)
        selected_sets.append(
            {
                "set_id": f"{strategy}_sample_{len(selected_sets):03d}",
                "set_kind": "sampled_disjoint",
                "sampling_strategy": strategy,
                "edges": list(edges),
            }
        )
    return selected_sets


def edge_set_membership_rows(
    edge_groups: Sequence[dict[str, Any]],
    *,
    set_id: str,
    set_kind: str,
    sampling_strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, edge in enumerate(sorted(edge_groups, key=edge_rank), start=1):
        rows.append(
            {
                "set_id": set_id,
                "set_kind": set_kind,
                "sampling_strategy": sampling_strategy,
                "edge_position": int(position),
                "rank": edge_rank(edge),
                "collapsed_edge": str(edge["collapsed_edge"]),
                "parent": str(edge["parent"]),
                "child": str(edge["child"]),
                "abs_score": float(edge["abs_score"]),
                "signed_sum": float(edge.get("signed_sum", edge["abs_score"])),
                "underlying_edge_count": int(edge.get("underlying_edge_count", len(edge.get("underlying_edges", [])))),
                "underlying_edges": "|".join(str(name) for name in edge.get("underlying_edges", [])),
            }
        )
    return rows


def evaluate_edge_set(
    *,
    model,
    base_graph,
    ranked_edges: Sequence[dict[str, Any]],
    validation_loader,
    metrics: dict[str, Any],
    edge_groups: Sequence[dict[str, Any]],
    set_id: str,
    set_kind: str,
    sampling_strategy: str,
    protected_budget: int,
    ablated_budget: int,
    candidate_start_rank: int,
    candidate_end_rank: int,
) -> dict[str, Any]:
    graph = build_ablate_circuit(base_graph, ranked_edges, list(edge_groups))
    row = evaluate_circuit(model, graph, validation_loader, metrics)
    row.update(
        {
            "set_id": set_id,
            "set_kind": set_kind,
            "sampling_strategy": sampling_strategy,
            "protected_budget": int(protected_budget),
            "ablated_budget": int(len(edge_groups)),
            "requested_ablated_budget": int(ablated_budget),
            "candidate_start_rank": int(candidate_start_rank),
            "candidate_end_rank": int(candidate_end_rank),
            "rank_min": min(edge_rank(edge) for edge in edge_groups),
            "rank_max": max(edge_rank(edge) for edge in edge_groups),
            "abs_score_sum": float(sum(edge_abs_score(edge) for edge in edge_groups)),
            "abs_score_mean": float(np.mean([edge_abs_score(edge) for edge in edge_groups])),
        }
    )
    del graph
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def summarize_mode(frame: pd.DataFrame, *, top_faithfulness: float) -> dict[str, Any]:
    if frame.empty:
        return {"count": 0}
    faithfulness = frame["faithfulness_mean"].astype(float)
    best_idx = int(faithfulness.idxmin())
    return {
        "count": int(len(frame)),
        "faithfulness_mean_mean": float(faithfulness.mean()),
        "faithfulness_mean_median": float(faithfulness.median()),
        "faithfulness_mean_min": float(faithfulness.min()),
        "faithfulness_mean_max": float(faithfulness.max()),
        "faithfulness_mean_p05": float(faithfulness.quantile(0.05)),
        "faithfulness_mean_p95": float(faithfulness.quantile(0.95)),
        "as_damaging_as_top_count": int((faithfulness <= float(top_faithfulness)).sum()),
        "as_damaging_as_top_rate": float((faithfulness <= float(top_faithfulness)).mean()),
        "best_set_id": str(frame.loc[best_idx, "set_id"]),
        "best_set_faithfulness_mean": float(frame.loc[best_idx, "faithfulness_mean"]),
    }


def summarize_results(result_frame: pd.DataFrame) -> dict[str, Any]:
    top_rows = result_frame[result_frame["set_kind"].astype(str) == "top_prefix"].copy()
    if top_rows.empty:
        raise ValueError("Conditional ablation results are missing the top-prefix baseline row.")
    top_row = top_rows.iloc[0]
    top_faithfulness = float(top_row["faithfulness_mean"])
    rank_bands = result_frame[result_frame["set_kind"].astype(str) == "rank_band"].copy()
    sampled = result_frame[result_frame["set_kind"].astype(str) == "sampled_disjoint"].copy()
    return {
        "top_prefix": {
            "set_id": str(top_row["set_id"]),
            "faithfulness_mean": top_faithfulness,
            "accuracy_mean": float(top_row["accuracy_mean"]),
            "kl_clean_mean": float(top_row["kl_clean_mean"]),
        },
        "rank_band": summarize_mode(rank_bands, top_faithfulness=top_faithfulness),
        "sampled_disjoint": summarize_mode(sampled, top_faithfulness=top_faithfulness),
    }


def resolve_analysis_config(
    project_root: Path,
    config: ConditionalAblationConfig,
) -> tuple[ConditionalAblationConfig, dict[str, Any] | None]:
    if config.localization_source_path is None:
        if config.edge_rankings_path is None:
            raise ValueError("Either localization_source_path or edge_rankings_path is required.")
        return config, None

    summary_path = resolve_localization_summary_path(
        project_root,
        config.localization_source_path,
        config.sample_size,
        config.seed,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_config = summary.get("config", {})
    dataset_summary = summary.get("dataset_summary", {})
    edge_rankings_value = summary.get("paths", {}).get("edge_rankings")
    if edge_rankings_value is None:
        raise ValueError(f"Localization summary does not contain an edge_rankings path: {summary_path}")

    config.model_name = str(summary_config.get("model_name", config.model_name))
    config.dataset_filter_model_name = str(
        summary_config.get("dataset_filter_model_name", config.dataset_filter_model_name)
    )
    config.discovery_margin_threshold = summary_config.get(
        "discovery_margin_threshold",
        config.discovery_margin_threshold,
    )
    summary_seeds = summary_config.get("seeds")
    if isinstance(summary_seeds, (list, tuple)) and summary_seeds:
        config.filtered_df_seed = min(int(seed) for seed in summary_seeds)
    else:
        config.filtered_df_seed = int(config.seed)
    config.dataset_filter_path = summary_config.get("dataset_filter_path", config.dataset_filter_path)
    config.cache_dataset_filter = bool(summary_config.get("cache_dataset_filter", config.cache_dataset_filter))
    config.max_filter_examples = summary_config.get("max_filter_examples", config.max_filter_examples)
    config.filter_batch_size = int(summary_config.get("filter_batch_size", config.filter_batch_size))
    config.target_filter_policy = str(summary_config.get("target_filter_policy", config.target_filter_policy))
    config.max_validation_examples = summary_config.get(
        "max_validation_examples",
        dataset_summary.get("max_validation_examples", config.max_validation_examples),
    )
    config.dataset_mode = str(dataset_summary.get("dataset_mode", summary_config.get("dataset_mode", config.dataset_mode)))
    config.dataset_set = str(summary_config.get("dataset_set", config.dataset_set))
    config.named_entity_discovery_dir = dataset_summary.get(
        "named_entity_discovery_dir",
        summary_config.get("named_entity_discovery_dir", config.named_entity_discovery_dir),
    )
    config.target_source = str(summary_config.get("target_source", config.target_source))
    config.target_token_mode = str(summary_config.get("target_token_mode", config.target_token_mode))
    config.edge_rankings_path = str(_resolve_path(project_root, edge_rankings_value))
    config.sample_size = int(summary["sample_size"])
    config.seed = int(summary["seed"])

    return config, {
        "summary_path": str(summary_path),
        "edge_rankings_path": str(config.edge_rankings_path),
        "discovery_sample_signature": dataset_summary.get("discovery_sample_signature"),
        "target_filtered_count": dataset_summary.get("target_filtered_count"),
        "validation_count": dataset_summary.get("validation_count"),
        "max_validation_examples": dataset_summary.get("max_validation_examples"),
        "filtered_df_seed": config.filtered_df_seed,
    }


def run_conditional_ablation_experiment(
    config: ConditionalAblationConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    config, localization_source = resolve_analysis_config(project_root, config)

    if config.band_size is None:
        config.band_size = int(config.ablated_budget)
    if config.protected_budget <= 0 or config.ablated_budget <= 0:
        raise ValueError("protected_budget and ablated_budget must be positive.")
    if config.band_size <= 0:
        raise ValueError("band_size must be positive.")
    if config.candidate_start_rank <= config.protected_budget:
        raise ValueError("candidate_start_rank must be greater than protected_budget.")
    if config.candidate_end_rank < config.candidate_start_rank:
        raise ValueError("candidate_end_rank must be >= candidate_start_rank.")
    if config.edge_rankings_path is None:
        raise ValueError("edge_rankings_path must be resolved before running conditional ablation.")

    day = config.output_day or date_tag()
    output_root = conditional_ablation_root(project_root, config.model_name, day)
    output_dir = ensure_dir(output_root / f"sample_{config.sample_size}" / f"seed_{config.seed}")

    localization_config = LocalizationConfig(
        model_name=config.model_name,
        dataset_filter_model_name=config.dataset_filter_model_name,
        discovery_margin_threshold=config.discovery_margin_threshold,
        filter_batch_size=config.filter_batch_size,
        evaluation_batch_size=config.evaluation_batch_size,
        max_validation_examples=config.max_validation_examples,
        output_day=config.output_day,
        dataset_filter_path=config.dataset_filter_path,
        refresh_dataset_filter=config.refresh_dataset_filter,
        cache_dataset_filter=config.cache_dataset_filter,
        max_filter_examples=config.max_filter_examples,
        target_filter_policy=config.target_filter_policy,
        dataset_mode=config.dataset_mode,
        dataset_set=config.dataset_set,
        named_entity_discovery_dir=config.named_entity_discovery_dir,
        target_source=config.target_source,
        target_token_mode=config.target_token_mode,
    )
    if config.dataset_mode == "named_entity_truncated":
        prepared = prepare_named_entity_truncated_inputs(project_root, localization_config)
    elif config.dataset_mode == "semantic_filtered":
        prepared = prepare_filtered_model_inputs(
            project_root=project_root,
            model_name=config.model_name,
            dataset_filter_model_name=config.dataset_filter_model_name,
            metric_batch_size=config.filter_batch_size,
            seed=config.filtered_df_seed if config.filtered_df_seed is not None else config.seed,
            dataset_filter_path=config.dataset_filter_path,
            refresh_dataset_filter=config.refresh_dataset_filter,
            cache_dataset_filter=config.cache_dataset_filter,
            max_filter_examples=config.max_filter_examples,
            target_filter_policy=config.target_filter_policy,
            target_source=config.target_source,
        )
    else:
        raise ValueError(f"Unsupported dataset mode: {config.dataset_mode}")

    discovery_df, validation_df, sample_signature = sample_discovery_validation(
        prepared["filtered_df"],
        discovery_sample_size=config.sample_size,
        seed=config.seed,
        discovery_margin_threshold=config.discovery_margin_threshold,
    )
    if localization_source is not None:
        expected_signature = localization_source.get("discovery_sample_signature")
        if expected_signature is not None and sample_signature != expected_signature:
            raise ValueError(
                "Reconstructed discovery sample does not match the saved localization slot: "
                f"{sample_signature!r} != {expected_signature!r}"
            )
        expected_filtered_count = localization_source.get("target_filtered_count")
        if expected_filtered_count is not None and int(len(prepared["filtered_df"])) != int(expected_filtered_count):
            raise ValueError(
                "Reconstructed filtered dataset size does not match the saved localization slot: "
                f"{len(prepared['filtered_df'])} != {expected_filtered_count}"
            )
    validation_df = maybe_limit_validation(
        validation_df,
        config.max_validation_examples,
        seed=config.seed + config.sample_size,
    )
    if localization_source is not None:
        expected_validation_count = localization_source.get("validation_count")
        if expected_validation_count is not None and int(len(validation_df)) != int(expected_validation_count):
            raise ValueError(
                "Reconstructed validation set size does not match the saved localization slot: "
                f"{len(validation_df)} != {expected_validation_count}"
            )
    validation_loader = make_dataloader(
        validation_df,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
    )

    metrics = make_eap_metrics(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )
    metrics["kl_to_clean"] = make_kl_to_clean_metric()

    ranked_edges = load_ranked_edges_csv(config.edge_rankings_path)
    if len(ranked_edges) < config.protected_budget:
        raise ValueError(
            f"Requested protected_budget={config.protected_budget} but ranking only has {len(ranked_edges)} edges."
        )
    base_graph = build_graph(prepared["model"])

    protected_edges = list(ranked_edges[: config.protected_budget])
    candidate_edges = filter_candidate_edges(
        ranked_edges,
        protected_budget=config.protected_budget,
        candidate_start_rank=config.candidate_start_rank,
        candidate_end_rank=config.candidate_end_rank,
    )
    if len(candidate_edges) < config.ablated_budget:
        raise ValueError(
            "Not enough disjoint candidate edges after filtering: "
            f"need {config.ablated_budget}, found {len(candidate_edges)}."
        )

    evaluation_sets = [
        {
            "set_id": f"top_{config.protected_budget}",
            "set_kind": "top_prefix",
            "sampling_strategy": "deterministic",
            "edges": protected_edges,
        },
        *build_rank_band_edge_sets(candidate_edges, set_size=config.band_size),
        *sample_edge_sets(
            candidate_edges,
            set_size=config.ablated_budget,
            sample_count=config.sample_count,
            strategy=config.sampling_strategy,
            random_seed=config.random_seed,
        ),
    ]

    result_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for edge_set in evaluation_sets:
        result_rows.append(
            evaluate_edge_set(
                model=prepared["model"],
                base_graph=base_graph,
                ranked_edges=ranked_edges,
                validation_loader=validation_loader,
                metrics=metrics,
                edge_groups=edge_set["edges"],
                set_id=edge_set["set_id"],
                set_kind=edge_set["set_kind"],
                sampling_strategy=edge_set["sampling_strategy"],
                protected_budget=config.protected_budget,
                ablated_budget=config.ablated_budget,
                candidate_start_rank=config.candidate_start_rank,
                candidate_end_rank=config.candidate_end_rank,
            )
        )
        membership_rows.extend(
            edge_set_membership_rows(
                edge_set["edges"],
                set_id=edge_set["set_id"],
                set_kind=edge_set["set_kind"],
                sampling_strategy=edge_set["sampling_strategy"],
            )
        )

    results_frame = pd.DataFrame(result_rows).sort_values(
        ["set_kind", "rank_min", "set_id"],
        kind="stable",
    ).reset_index(drop=True)
    memberships_frame = pd.DataFrame(membership_rows).sort_values(
        ["set_kind", "set_id", "edge_position"],
        kind="stable",
    ).reset_index(drop=True)
    protected_frame = pd.DataFrame(
        edge_set_membership_rows(
            protected_edges,
            set_id=f"protected_top_{config.protected_budget}",
            set_kind="protected_prefix",
            sampling_strategy="deterministic",
        )
    )
    summary = summarize_results(results_frame)

    results_path = output_dir / "conditional_ablation_results.csv"
    memberships_path = output_dir / "conditional_ablation_sets.csv"
    protected_path = output_dir / "protected_top_edges.csv"
    summary_path = output_dir / f"conditional_ablation_summary_{timestamp_tag()}.json"
    save_csv(results_frame, results_path, index=False)
    save_csv(memberships_frame, memberships_path, index=False)
    save_csv(protected_frame, protected_path, index=False)

    manifest = {
        "experiment": "eap_ig_conditional_ablation",
        "config": asdict(config),
        "paths": {
            "project_root": str(project_root),
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "results": str(results_path),
            "set_memberships": str(memberships_path),
            "protected_edges": str(protected_path),
            "summary": str(summary_path),
        },
        "dataset_summary": {
            "dataset_mode": config.dataset_mode,
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "source_filter_model": prepared["dataset_filter_model_name"],
            "target_filtered_count": int(len(prepared["filtered_df"])),
            "discovery_count": int(len(discovery_df)),
            "validation_count": int(len(validation_df)),
            "max_validation_examples": config.max_validation_examples,
            "discovery_sample_signature": sample_signature,
        },
        "ranking_source": localization_source or {"edge_rankings_path": str(config.edge_rankings_path)},
        "candidate_pool": {
            "candidate_count": int(len(candidate_edges)),
            "candidate_start_rank": int(config.candidate_start_rank),
            "candidate_end_rank": int(config.candidate_end_rank),
            "band_count": int(sum(1 for edge_set in evaluation_sets if edge_set["set_kind"] == "rank_band")),
            "sampled_set_count": int(sum(1 for edge_set in evaluation_sets if edge_set["set_kind"] == "sampled_disjoint")),
        },
        "summary": summary,
    }
    save_json(summary_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run conditional disjoint-set ablations against a saved localization ranking."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset-filter-model", default="gpt2")
    parser.add_argument("--discovery-margin-threshold", type=float, default=DEFAULT_DISCOVERY_MARGIN_THRESHOLD)
    parser.add_argument("--filter-batch-size", type=int, default=50)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--output-day", default=None)
    parser.add_argument("--dataset-filter-path", default=None)
    parser.add_argument("--refresh-dataset-filter", action="store_true")
    parser.add_argument(
        "--cache-dataset-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-filter-examples", type=int, default=None)
    parser.add_argument(
        "--target-filter-policy",
        choices=("none", "recovery_margin", "model_success"),
        default="model_success",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("semantic_filtered", "named_entity_truncated"),
        default="semantic_filtered",
    )
    parser.add_argument("--dataset-set", default=MODEL_SPECIFIC_CORRECT)
    parser.add_argument("--named-entity-discovery-dir", default=None)
    parser.add_argument("--target-source", default="dataset/semantic_meaningful/named_entity_targets.json")
    parser.add_argument("--target-token-mode", default="first_token")
    parser.add_argument("--localization-source-path", default=None)
    parser.add_argument("--edge-rankings-path", default=None)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protected-budget", type=int, default=20)
    parser.add_argument("--ablated-budget", type=int, default=20)
    parser.add_argument("--candidate-start-rank", type=int, default=21)
    parser.add_argument("--candidate-end-rank", type=int, default=200)
    parser.add_argument("--band-size", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--sampling-strategy", choices=("score_weighted", "uniform"), default="score_weighted")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--start-path", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ConditionalAblationConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        discovery_margin_threshold=args.discovery_margin_threshold,
        filter_batch_size=args.filter_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        max_validation_examples=args.max_validation_examples,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        dataset_mode=args.dataset_mode,
        dataset_set=args.dataset_set,
        named_entity_discovery_dir=args.named_entity_discovery_dir,
        target_source=args.target_source,
        target_token_mode=args.target_token_mode,
        localization_source_path=args.localization_source_path,
        edge_rankings_path=args.edge_rankings_path,
        sample_size=args.sample_size,
        seed=args.seed,
        protected_budget=args.protected_budget,
        ablated_budget=args.ablated_budget,
        candidate_start_rank=args.candidate_start_rank,
        candidate_end_rank=args.candidate_end_rank,
        band_size=args.band_size,
        sample_count=args.sample_count,
        sampling_strategy=args.sampling_strategy,
        random_seed=args.random_seed,
    )
    manifest = run_conditional_ablation_experiment(config=config, start=args.start_path)
    print(f"Saved conditional ablation outputs to {manifest['paths']['output_dir']}")
    print(f"Summary: {manifest['paths']['summary']}")


if __name__ == "__main__":
    main()

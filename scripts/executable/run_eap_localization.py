from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from circuit_finder_core import (
    DEFAULT_DISCOVERY_MARGIN_THRESHOLD,
    DEFAULT_EAP_BUDGET_MAX_FRACTION,
    DEFAULT_EAP_BUDGET_TAIL_POINTS,
    MODEL_SPECIFIC_CORRECT,
    build_graph,
    canonical_model_name,
    clone_graph,
    collapsed_edge_groups,
    date_tag,
    eap_node_metadata,
    ensure_dir,
    induced_node_ranking,
    load_model,
    make_dataloader,
    make_eap_metrics,
    prepare_filtered_model_inputs,
    ranking_frame,
    resolve_animacy_circuit_root,
    resolve_eap_budget_grid,
    safe_model_name,
    sample_discovery_validation,
    save_csv,
    save_json,
    timestamp_tag,
)


DEFAULT_TOP_K = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
DEFAULT_SAMPLE_SIZES = (100, 250, 500, 1000, 2000)
DEFAULT_SEEDS = (0, 1, 42)
DEFAULT_RANDOM_REPEATS = 10
CONCENTRATION_TOP_K = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
MASS_THRESHOLDS = (0.5, 0.8, 0.9, 0.95)


@dataclass
class LocalizationConfig:
    model_name: str = "gpt2"
    dataset_filter_model_name: str = "gpt2"
    sample_sizes: tuple[int, ...] = DEFAULT_SAMPLE_SIZES
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    top_k: tuple[int, ...] = DEFAULT_TOP_K
    random_repeats: int = DEFAULT_RANDOM_REPEATS
    discovery_margin_threshold: float | None = DEFAULT_DISCOVERY_MARGIN_THRESHOLD
    filter_batch_size: int = 50
    attribution_batch_size: int = 8
    evaluation_batch_size: int = 1
    ig_steps: int = 5
    max_validation_examples: int | None = None
    budget_max_fraction: float = DEFAULT_EAP_BUDGET_MAX_FRACTION
    budget_floor: int = 2000
    budget_tail_points: int = DEFAULT_EAP_BUDGET_TAIL_POINTS
    output_day: str | None = None
    dataset_filter_path: str | None = None
    refresh_dataset_filter: bool = False
    cache_dataset_filter: bool = True
    max_filter_examples: int | None = None
    target_filter_policy: str = "model_success"
    skip_random_baselines: bool = False
    skip_existing: bool = False
    import_edge_rankings_path: str | None = None
    import_summary_path: str | None = None
    import_sample_size: int | None = None
    import_seed: int | None = None
    dataset_mode: str = "semantic_filtered"
    dataset_set: str = MODEL_SPECIFIC_CORRECT
    named_entity_discovery_dir: str | None = None
    target_source: str = "dataset/semantic_meaningful/named_entity_targets.json"
    target_token_mode: str = "first_token"


def localization_root(project_root: Path, model_name: str, day: str | None = None) -> Path:
    resolved = canonical_model_name(model_name)
    return ensure_dir(
        project_root
        / "results"
        / "eap_ig_localization"
        / safe_model_name(resolved)
        / (date_tag() if day is None else day)
    )


def run_output_dir(base_dir: Path, sample_size: int, seed: int) -> Path:
    return ensure_dir(base_dir / f"sample_{sample_size}" / f"seed_{seed}")


def _split_underlying_edges(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if pd.isna(value):
        return []
    return [part for part in str(value).split("|") if part]


def load_ranked_edges_csv(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"collapsed_edge", "parent", "child", "abs_score", "underlying_edges"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Imported edge ranking {path} is missing required columns: {missing}")
    if "rank" in frame.columns:
        ordered = frame.sort_values("rank", ascending=True, kind="stable").reset_index(drop=True)
    else:
        ordered = frame.sort_values("abs_score", ascending=False, kind="stable").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ordered.to_dict(orient="records"), start=1):
        rows.append(
            {
                "collapsed_edge": str(row["collapsed_edge"]),
                "parent": str(row["parent"]),
                "child": str(row["child"]),
                "signed_sum": float(row.get("signed_sum", row["abs_score"])),
                "abs_score": float(row["abs_score"]),
                "underlying_edges": _split_underlying_edges(row["underlying_edges"]),
                "rank": int(row.get("rank", idx)),
                "underlying_edge_count": int(
                    row.get("underlying_edge_count", len(_split_underlying_edges(row["underlying_edges"])))
                ),
            }
        )
    return rows


def infer_import_summary_path(edge_rankings_path: Path | str) -> Path | None:
    edge_rankings_path = Path(edge_rankings_path)
    candidates = sorted(edge_rankings_path.parent.glob("full_model_summary_*.json"))
    candidates.extend(sorted(edge_rankings_path.parent.glob("named_entity_discovery_summary_*.json")))
    candidates = sorted(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return candidates[-1] if candidates else None


def validate_imported_ranking(
    *,
    summary_path: Path | None,
    config: LocalizationConfig,
    prepared: dict[str, Any],
    sample_size: int,
    seed: int,
    sample_signature: dict[str, Any],
) -> dict[str, Any]:
    if summary_path is None:
        return {"status": "unverified", "reason": "no_summary"}
    if not summary_path.is_file():
        raise FileNotFoundError(f"Import summary path does not exist: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") == "named_entity_full_model_discovery":
        summary_config = summary.get("config", {})
        dataset_counts = summary.get("dataset_counts", {})
        mismatches: list[str] = []

        if canonical_model_name(summary.get("model", config.model_name)) != canonical_model_name(config.model_name):
            mismatches.append(f"model={summary.get('model')!r} != {canonical_model_name(config.model_name)!r}")
        if int(dataset_counts.get("discovery", sample_size)) != int(sample_size):
            mismatches.append(f"discovery={dataset_counts.get('discovery')!r} != {sample_size}")
        if int(summary_config.get("seed", seed)) != int(seed):
            mismatches.append(f"seed={summary_config.get('seed')!r} != {seed}")
        if summary_config.get("discovery_sample_size") is not None and int(summary_config["discovery_sample_size"]) != int(sample_size):
            mismatches.append(
                f"discovery_sample_size={summary_config.get('discovery_sample_size')!r} != {sample_size}"
            )
        if dataset_counts.get("discovery_sample_signature") != sample_signature:
            mismatches.append("discovery_sample_signature mismatch")

        if mismatches:
            raise ValueError(
                "Imported named-entity edge ranking does not match the requested localization slot: "
                + "; ".join(mismatches)
            )

        return {
            "status": "verified",
            "summary_path": str(summary_path),
            "experiment": summary.get("experiment"),
        }

    dataset_summary = summary.get("dataset_summary", {})
    summary_config = summary.get("config", {})
    mismatches: list[str] = []

    if canonical_model_name(dataset_summary.get("target_model", config.model_name)) != canonical_model_name(config.model_name):
        mismatches.append(
            f"target_model={dataset_summary.get('target_model')!r} != {canonical_model_name(config.model_name)!r}"
        )
    if canonical_model_name(
        dataset_summary.get("source_filter_model", config.dataset_filter_model_name)
    ) != canonical_model_name(config.dataset_filter_model_name):
        mismatches.append(
            "source_filter_model="
            f"{dataset_summary.get('source_filter_model')!r} != {canonical_model_name(config.dataset_filter_model_name)!r}"
        )
    if dataset_summary.get("target_filter_policy", config.target_filter_policy) != config.target_filter_policy:
        mismatches.append(
            f"target_filter_policy={dataset_summary.get('target_filter_policy')!r} != {config.target_filter_policy!r}"
        )
    if int(dataset_summary.get("discovery_count", sample_size)) != int(sample_size):
        mismatches.append(f"discovery_count={dataset_summary.get('discovery_count')!r} != {sample_size}")
    if int(summary_config.get("seed", seed)) != int(seed):
        mismatches.append(f"seed={summary_config.get('seed')!r} != {seed}")
    if summary_config.get("discovery_sample_size") is not None and int(summary_config["discovery_sample_size"]) != int(sample_size):
        mismatches.append(
            f"discovery_sample_size={summary_config.get('discovery_sample_size')!r} != {sample_size}"
        )
    if dataset_summary.get("discovery_sample_signature") != sample_signature:
        mismatches.append("discovery_sample_signature mismatch")

    if mismatches:
        raise ValueError(
            "Imported edge ranking does not match the requested localization slot: "
            + "; ".join(mismatches)
        )

    return {
        "status": "verified",
        "summary_path": str(summary_path),
        "experiment": summary.get("experiment"),
    }


def _resolve_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == project_root.name:
        return project_root.parent / path
    return project_root / path


def resolve_named_entity_discovery_dir(
    project_root: Path,
    config: LocalizationConfig,
) -> Path:
    if config.named_entity_discovery_dir is not None:
        path = _resolve_path(project_root, config.named_entity_discovery_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Named-entity discovery directory does not exist: {path}")
        return path

    model_slug = safe_model_name(canonical_model_name(config.model_name))
    base = project_root / "results" / "named_entity_discovery" / model_slug / config.dataset_set
    candidates = sorted(
        [path for path in base.glob("named_entity_discovery_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No named-entity discovery directories found under {base}")
    return candidates[0]


def latest_matching_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} found under {directory}")
    return candidates[-1]


def named_entity_accuracy_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"example_count": int(len(frame))}
    if "clean_metric" in frame.columns:
        summary["clean_accuracy"] = float((frame["clean_metric"] > 0.0).mean()) if len(frame) else 0.0
        summary["clean_metric_mean"] = float(frame["clean_metric"].mean()) if len(frame) else 0.0
    if "corrupt_metric" in frame.columns:
        summary["corrupt_accuracy"] = float((frame["corrupt_metric"] < 0.0).mean()) if len(frame) else 0.0
        summary["corrupt_metric_mean"] = float(frame["corrupt_metric"].mean()) if len(frame) else 0.0
    if {"clean_metric", "corrupt_metric"}.issubset(frame.columns):
        summary["model_success_rate"] = float(
            (
                (frame["clean_metric"] > 0.0)
                & (frame["corrupt_metric"] < 0.0)
                & ((frame["clean_metric"] - frame["corrupt_metric"]) > 1e-6)
            ).mean()
        ) if len(frame) else 0.0
    return summary


def prepare_named_entity_truncated_inputs(
    project_root: Path,
    config: LocalizationConfig,
) -> dict[str, Any]:
    from evaluate_named_entity_circuit import named_entity_target_tensors

    model_name = canonical_model_name(config.model_name)
    discovery_dir = resolve_named_entity_discovery_dir(project_root, config)
    summary_path = latest_matching_file(discovery_dir, "named_entity_discovery_summary_*.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = summary.get("paths", {})

    retained_path_value = paths.get("retained_dataset") or latest_matching_file(
        discovery_dir,
        "named_entity_model_success_*.csv",
    )
    scored_path_value = paths.get("scored_dataset") or latest_matching_file(
        discovery_dir,
        "named_entity_truncated_scored_*.csv",
    )
    retained_path = _resolve_path(project_root, retained_path_value)
    scored_path = _resolve_path(project_root, scored_path_value)

    print(f"Loading named-entity retained dataset from {retained_path}")
    retained_df = pd.read_csv(retained_path)
    scored_df = pd.read_csv(scored_path) if scored_path.is_file() else retained_df.copy()

    print(f"Loading target model {model_name}.")
    model = load_model(model_name)
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError(f"Model {model_name} has no tokenizer attached.")

    target_source = summary.get("target_source", config.target_source)
    target_token_mode = summary.get("target_token_mode", config.target_token_mode)
    animate_ids, inanimate_ids, target_summary, target_path = named_entity_target_tensors(
        project_root,
        target_source,
        tokenizer,
        model.cfg.device,
        target_token_mode,
    )

    return {
        "source_success_df": retained_df,
        "target_raw_scored_df": scored_df,
        "target_scored_df": retained_df,
        "filtered_df": retained_df,
        "raw_tokenization_diagnostics": {},
        "target_raw_accuracy": named_entity_accuracy_summary(scored_df),
        "target_on_source_accuracy": named_entity_accuracy_summary(retained_df),
        "source_success_cache_path": str(retained_path),
        "source_success_cache_status": "loaded_named_entity_discovery",
        "requested_model_name": config.model_name,
        "model_name": model_name,
        "requested_dataset_filter_model_name": config.dataset_filter_model_name,
        "dataset_filter_model_name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "animate_ids_tensor": animate_ids,
        "inanimate_ids_tensor": inanimate_ids,
        "target_filter_summary": target_summary,
        "target_source_path": str(target_path),
        "named_entity_discovery_dir": str(discovery_dir),
        "named_entity_discovery_summary": str(summary_path),
    }


def _score_values(ranked_edges: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.array([abs(float(edge["abs_score"])) for edge in ranked_edges], dtype=float)


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    if np.min(values) < 0:
        values = values - np.min(values)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values) / (n * total)) - ((n + 1) / n))


def concentration_summary(
    ranked_edges: Sequence[dict[str, Any]],
    top_k_values: Sequence[int] = CONCENTRATION_TOP_K,
) -> dict[str, Any]:
    values = _score_values(ranked_edges)
    total = float(values.sum())
    n_edges = int(values.size)
    if n_edges == 0 or total <= 0:
        row: dict[str, Any] = {
            "ranked_edge_count": n_edges,
            "total_abs_score": total,
            "gini": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_edge_count": 0.0,
        }
        for k in top_k_values:
            row[f"top_{k}_mass"] = 0.0
        for threshold in MASS_THRESHOLDS:
            row[f"edges_for_{int(threshold * 100)}pct_mass"] = 0
        return row

    probs = values / total
    entropy = float(-(probs * np.log(probs + 1e-30)).sum())
    cumulative = np.cumsum(values) / total
    row = {
        "ranked_edge_count": n_edges,
        "total_abs_score": total,
        "gini": gini(values),
        "entropy": entropy,
        "normalized_entropy": float(entropy / math.log(n_edges)) if n_edges > 1 else 0.0,
        "effective_edge_count": float(1.0 / np.square(probs).sum()),
    }
    for k in top_k_values:
        effective_k = min(int(k), n_edges)
        row[f"top_{k}_mass"] = float(cumulative[effective_k - 1]) if effective_k else 0.0
    for threshold in MASS_THRESHOLDS:
        row[f"edges_for_{int(threshold * 100)}pct_mass"] = int(
            np.searchsorted(cumulative, threshold, side="left") + 1
        )
    return row


def cumulative_mass_frame(ranked_edges: Sequence[dict[str, Any]]) -> pd.DataFrame:
    values = _score_values(ranked_edges)
    total = float(values.sum())
    if len(values) == 0:
        return pd.DataFrame(columns=["rank", "collapsed_edge", "abs_score", "cumulative_mass"])
    cumulative = np.cumsum(values) / total if total > 0 else np.zeros_like(values)
    return pd.DataFrame(
        {
            "rank": [int(edge["rank"]) for edge in ranked_edges],
            "collapsed_edge": [edge["collapsed_edge"] for edge in ranked_edges],
            "abs_score": values,
            "cumulative_mass": cumulative,
        }
    )


def maybe_limit_validation(
    validation_df: pd.DataFrame,
    max_validation_examples: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_validation_examples is None or max_validation_examples >= len(validation_df):
        return validation_df.reset_index(drop=True).copy()
    if max_validation_examples <= 0:
        raise ValueError("max_validation_examples must be positive when provided.")
    return (
        validation_df.sample(n=max_validation_examples, random_state=seed)
        .reset_index(drop=True)
        .copy()
    )


def build_keep_circuit(scored_graph, edge_groups: Sequence[dict[str, Any]]):
    candidate = clone_graph(scored_graph)
    candidate.reset()
    for edge_group in edge_groups:
        for edge_name in edge_group["underlying_edges"]:
            candidate.edges[edge_name].in_graph = True
    candidate.prune()
    return candidate


def build_ablate_circuit(
    scored_graph,
    ranked_edges: Sequence[dict[str, Any]],
    ablated_edge_groups: Sequence[dict[str, Any]],
):
    ablated = {
        edge_name
        for edge_group in ablated_edge_groups
        for edge_name in edge_group["underlying_edges"]
    }
    candidate = clone_graph(scored_graph)
    candidate.reset()
    for edge_name, edge in candidate.edges.items():
        if not bool(candidate.real_edge_mask[edge.matrix_index].item()):
            continue
        if edge_name not in ablated:
            edge.in_graph = True
    candidate.prune()
    return candidate


def make_kl_to_clean_metric():
    def metric(
        logits: torch.Tensor,
        clean_logits: torch.Tensor | None,
        input_lengths: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        del input_lengths, label
        if clean_logits is None:
            raise ValueError("clean_logits is required to compute KL-to-clean.")
        patched_log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        clean_probs = F.softmax(clean_logits[:, -1, :], dim=-1)
        return F.kl_div(patched_log_probs, clean_probs, reduction="none").sum(dim=-1)

    return metric


def evaluate_circuit(
    model,
    graph,
    validation_loader,
    metrics: dict[str, Callable[..., torch.Tensor]],
) -> dict[str, Any]:
    from eap.evaluate import evaluate_graph

    faithfulness_values, accuracy_values, kl_values = evaluate_graph(
        model,
        graph,
        validation_loader,
        [metrics["faithfulness"], metrics["accuracy"], metrics["kl_to_clean"]],
        quiet=True,
        intervention="patching",
        skip_clean=False,
    )
    faithfulness = faithfulness_values.float().detach().cpu()
    accuracy = accuracy_values.float().detach().cpu()
    kl = kl_values.float().detach().cpu()
    return {
        "faithfulness_mean": float(faithfulness.mean().item()),
        "faithfulness_std": float(faithfulness.std(unbiased=False).item()) if len(faithfulness) > 1 else 0.0,
        "accuracy_mean": float(accuracy.mean().item()),
        "accuracy_std": float(accuracy.std(unbiased=False).item()) if len(accuracy) > 1 else 0.0,
        "kl_clean_mean": float(kl.mean().item()),
        "kl_clean_std": float(kl.std(unbiased=False).item()) if len(kl) > 1 else 0.0,
        "validation_examples": int(len(faithfulness)),
        "expanded_edge_count": int(graph.count_included_edges()),
        "induced_node_count": int(graph.count_included_nodes() - 2),
    }


def node_signature(node_name: str, include_layer: bool = True) -> tuple[Any, ...]:
    meta = eap_node_metadata(node_name)
    if include_layer:
        return meta["kind"], meta["layer"]
    return (meta["kind"],)


def edge_signature(edge_group: dict[str, Any], include_layer: bool = True) -> tuple[Any, ...]:
    return (
        *node_signature(edge_group["parent"], include_layer=include_layer),
        "->",
        *node_signature(edge_group["child"], include_layer=include_layer),
    )


def choose_matched_random_edges(
    ranked_edges: Sequence[dict[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    target_edges = list(ranked_edges[:budget])
    excluded = {edge["collapsed_edge"] for edge in target_edges}
    candidates = [edge for edge in ranked_edges if edge["collapsed_edge"] not in excluded]
    if len(candidates) < budget:
        candidates = list(ranked_edges)

    strict_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    broad_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for edge in candidates:
        strict_buckets.setdefault(edge_signature(edge, include_layer=True), []).append(edge)
        broad_buckets.setdefault(edge_signature(edge, include_layer=False), []).append(edge)

    chosen: list[dict[str, Any]] = []
    chosen_names: set[str] = set()

    def take_from(pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        available = [edge for edge in pool if edge["collapsed_edge"] not in chosen_names]
        if not available:
            return None
        return rng.choice(available)

    for target in target_edges:
        selected = take_from(strict_buckets.get(edge_signature(target, True), []))
        if selected is None:
            selected = take_from(broad_buckets.get(edge_signature(target, False), []))
        if selected is None:
            selected = take_from(candidates)
        if selected is None:
            break
        chosen.append(selected)
        chosen_names.add(selected["collapsed_edge"])

    return chosen


def evaluate_topk_and_random(
    model,
    scored_graph,
    ranked_edges: Sequence[dict[str, Any]],
    validation_loader,
    metrics: dict[str, Callable[..., torch.Tensor]],
    top_k_values: Sequence[int],
    random_repeats: int,
    seed: int,
    skip_random_baselines: bool,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    effective_top_k = [int(k) for k in top_k_values if 0 < int(k) <= len(ranked_edges)]

    for budget in tqdm(effective_top_k, desc="Top-k diagnostics"):
        top_groups = list(ranked_edges[:budget])
        for mode, graph in (
            ("keep_top", build_keep_circuit(scored_graph, top_groups)),
            ("ablate_top", build_ablate_circuit(scored_graph, ranked_edges, top_groups)),
        ):
            row = evaluate_circuit(model, graph, validation_loader, metrics)
            row.update(
                {
                    "mode": mode,
                    "baseline": "eap_ranked",
                    "collapsed_edge_budget": int(budget),
                    "repeat": 0,
                    "matched_random": False,
                }
            )
            rows.append(row)
            if checkpoint_path is not None:
                save_csv(pd.DataFrame(rows), checkpoint_path, index=False)

        if skip_random_baselines:
            continue

        for repeat in range(random_repeats):
            rng = random.Random((seed + 1) * 1_000_003 + budget * 1_009 + repeat)
            random_groups = choose_matched_random_edges(ranked_edges, budget, rng)
            if len(random_groups) != budget:
                continue
            for mode, graph in (
                ("keep_random", build_keep_circuit(scored_graph, random_groups)),
                ("ablate_random", build_ablate_circuit(scored_graph, ranked_edges, random_groups)),
            ):
                row = evaluate_circuit(model, graph, validation_loader, metrics)
                row.update(
                    {
                        "mode": mode,
                        "baseline": "layer_type_matched_random",
                        "collapsed_edge_budget": int(budget),
                        "repeat": int(repeat),
                        "matched_random": True,
                    }
                )
                rows.append(row)
                if checkpoint_path is not None:
                    save_csv(pd.DataFrame(rows), checkpoint_path, index=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def edge_stability(
    run_edges: dict[tuple[int, int], pd.DataFrame],
    top_k_values: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_size in sorted({key[0] for key in run_edges}):
        seeds = sorted(seed for size, seed in run_edges if size == sample_size)
        for seed_a, seed_b in combinations(seeds, 2):
            edges_a = run_edges[(sample_size, seed_a)].copy()
            edges_b = run_edges[(sample_size, seed_b)].copy()
            score_a = edges_a.set_index("collapsed_edge")["abs_score"].astype(float)
            score_b = edges_b.set_index("collapsed_edge")["abs_score"].astype(float)
            union = sorted(set(score_a.index) | set(score_b.index))
            rank_frame = pd.DataFrame(
                {
                    "a": score_a.reindex(union).fillna(0.0),
                    "b": score_b.reindex(union).fillna(0.0),
                }
            )
            # Avoid a hard SciPy dependency for aggregation-only stability metrics.
            rank_a = rank_frame["a"].rank(method="average")
            rank_b = rank_frame["b"].rank(method="average")
            spearman_value = rank_a.corr(rank_b, method="pearson")
            spearman = float(spearman_value) if pd.notna(spearman_value) else 0.0
            for k in top_k_values:
                top_a = set(edges_a.head(k)["collapsed_edge"])
                top_b = set(edges_b.head(k)["collapsed_edge"])
                if not top_a and not top_b:
                    jaccard = 0.0
                    overlap = 0
                else:
                    overlap = len(top_a & top_b)
                    jaccard = float(overlap / len(top_a | top_b))
                rows.append(
                    {
                        "sample_size": int(sample_size),
                        "seed_a": int(seed_a),
                        "seed_b": int(seed_b),
                        "top_k": int(k),
                        "edge_overlap": int(overlap),
                        "edge_jaccard": jaccard,
                        "edge_overlap_rate": float(overlap / min(len(top_a), len(top_b)))
                        if min(len(top_a), len(top_b))
                        else 0.0,
                        "spearman_abs_score": spearman,
                    }
                )
    columns = [
        "sample_size",
        "seed_a",
        "seed_b",
        "top_k",
        "edge_overlap",
        "edge_jaccard",
        "edge_overlap_rate",
        "spearman_abs_score",
    ]
    return pd.DataFrame(rows, columns=columns)


def node_stability(
    run_nodes: dict[tuple[int, int], pd.DataFrame],
    top_k_values: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_size in sorted({key[0] for key in run_nodes}):
        seeds = sorted(seed for size, seed in run_nodes if size == sample_size)
        for seed_a, seed_b in combinations(seeds, 2):
            nodes_a = run_nodes[(sample_size, seed_a)].copy()
            nodes_b = run_nodes[(sample_size, seed_b)].copy()
            for k in top_k_values:
                top_a = set(nodes_a.head(k)["node"])
                top_b = set(nodes_b.head(k)["node"])
                overlap = len(top_a & top_b)
                rows.append(
                    {
                        "sample_size": int(sample_size),
                        "seed_a": int(seed_a),
                        "seed_b": int(seed_b),
                        "top_k": int(k),
                        "node_overlap": int(overlap),
                        "node_jaccard": float(overlap / len(top_a | top_b))
                        if (top_a or top_b)
                        else 0.0,
                        "node_overlap_rate": float(overlap / min(len(top_a), len(top_b)))
                        if min(len(top_a), len(top_b))
                        else 0.0,
                    }
                )
    columns = [
        "sample_size",
        "seed_a",
        "seed_b",
        "top_k",
        "node_overlap",
        "node_jaccard",
        "node_overlap_rate",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_localization_experiment(
    config: LocalizationConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    day = config.output_day or date_tag()
    output_root = localization_root(project_root, config.model_name, day)

    if config.dataset_mode == "named_entity_truncated":
        prepared = prepare_named_entity_truncated_inputs(project_root, config)
    elif config.dataset_mode == "semantic_filtered":
        prepared = prepare_filtered_model_inputs(
            project_root=project_root,
            model_name=config.model_name,
            dataset_filter_model_name=config.dataset_filter_model_name,
            metric_batch_size=config.filter_batch_size,
            seed=min(config.seeds),
            dataset_filter_path=config.dataset_filter_path,
            refresh_dataset_filter=config.refresh_dataset_filter,
            cache_dataset_filter=config.cache_dataset_filter,
            max_filter_examples=config.max_filter_examples,
            target_filter_policy=config.target_filter_policy,
        )
    else:
        raise ValueError(f"Unsupported dataset mode: {config.dataset_mode}")
    model = prepared["model"]
    metrics = make_eap_metrics(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )
    metrics["kl_to_clean"] = make_kl_to_clean_metric()

    if config.import_edge_rankings_path is not None:
        if config.import_sample_size is None or config.import_seed is None:
            raise ValueError(
                "import_sample_size and import_seed are required when import_edge_rankings_path is provided."
            )
        slot = (int(config.import_sample_size), int(config.import_seed))
        available_slots = {(int(sample_size), int(seed)) for sample_size in config.sample_sizes for seed in config.seeds}
        if slot not in available_slots:
            raise ValueError(
                f"Imported slot {slot} is not in the requested localization grid {sorted(available_slots)}."
            )
        imported_ranking_path = Path(config.import_edge_rankings_path)
        if not imported_ranking_path.is_file():
            raise FileNotFoundError(f"Imported edge ranking path does not exist: {imported_ranking_path}")
        imported_summary_path = (
            Path(config.import_summary_path)
            if config.import_summary_path is not None
            else infer_import_summary_path(imported_ranking_path)
        )
    else:
        imported_ranking_path = None
        imported_summary_path = None

    run_summaries: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    run_edges: dict[tuple[int, int], pd.DataFrame] = {}
    run_nodes: dict[tuple[int, int], pd.DataFrame] = {}

    for sample_size in config.sample_sizes:
        for seed in config.seeds:
            output_dir = run_output_dir(output_root, sample_size, seed)
            summary_path = output_dir / f"localization_summary_sample_{sample_size}_seed_{seed}.json"
            if config.skip_existing and summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                run_summaries.append(summary)
                if "concentration" in summary:
                    concentration_rows.append(summary["concentration"])
                edge_path = Path(summary["paths"]["edge_rankings"])
                node_path = Path(summary["paths"]["node_rankings"])
                if edge_path.is_file():
                    run_edges[(sample_size, seed)] = pd.read_csv(edge_path)
                if node_path.is_file():
                    run_nodes[(sample_size, seed)] = pd.read_csv(node_path)
                continue

            discovery_df, validation_df, sample_signature = sample_discovery_validation(
                prepared["filtered_df"],
                discovery_sample_size=sample_size,
                seed=seed,
                discovery_margin_threshold=config.discovery_margin_threshold,
            )
            validation_df = maybe_limit_validation(
                validation_df,
                config.max_validation_examples,
                seed=seed + sample_size,
            )
            validation_loader = make_dataloader(
                validation_df,
                batch_size=config.evaluation_batch_size,
                shuffle=False,
            )
            import_metadata: dict[str, Any] | None = None
            if imported_ranking_path is not None and sample_size == config.import_sample_size and seed == config.import_seed:
                ranked_edges = load_ranked_edges_csv(imported_ranking_path)
                scored_graph = build_graph(model)
                import_metadata = validate_imported_ranking(
                    summary_path=imported_summary_path,
                    config=config,
                    prepared=prepared,
                    sample_size=sample_size,
                    seed=seed,
                    sample_signature=sample_signature,
                )
            else:
                discovery_loader = make_dataloader(
                    discovery_df,
                    batch_size=config.attribution_batch_size,
                    shuffle=False,
                )
                scored_graph = build_graph(model)
                from circuit_finder_core import attribute_graph

                scored_graph = attribute_graph(
                    model=model,
                    graph=scored_graph,
                    dataloader=discovery_loader,
                    metric=metrics["attribute"],
                    ig_steps=config.ig_steps,
                )
                ranked_edges = collapsed_edge_groups(scored_graph)
            ranked_nodes = induced_node_ranking(ranked_edges)
            edge_frame = ranking_frame(ranked_edges)
            node_frame = ranking_frame(ranked_nodes)
            run_edges[(sample_size, seed)] = edge_frame
            run_nodes[(sample_size, seed)] = node_frame

            top_k = resolve_eap_budget_grid(
                len(ranked_edges),
                budgets=config.top_k,
                budget_max_fraction=config.budget_max_fraction,
                budget_floor=config.budget_floor,
                budget_tail_points=config.budget_tail_points,
            )
            concentration = concentration_summary(ranked_edges)
            concentration.update({"sample_size": int(sample_size), "seed": int(seed)})
            concentration_rows.append(concentration)
            cumulative = cumulative_mass_frame(ranked_edges)
            cumulative.insert(0, "sample_size", int(sample_size))
            cumulative.insert(1, "seed", int(seed))

            edge_path = output_dir / f"edges_sample_{sample_size}_seed_{seed}.csv"
            node_path = output_dir / f"nodes_sample_{sample_size}_seed_{seed}.csv"
            eval_path = output_dir / f"topk_evaluations_sample_{sample_size}_seed_{seed}.csv"
            eval_checkpoint_path = output_dir / f"topk_evaluations_partial_sample_{sample_size}_seed_{seed}.csv"
            cumulative_path = output_dir / f"cumulative_mass_sample_{sample_size}_seed_{seed}.csv"
            save_csv(edge_frame, edge_path, index=False)
            save_csv(node_frame, node_path, index=False)
            save_csv(cumulative, cumulative_path, index=False)
            evaluations = evaluate_topk_and_random(
                model=model,
                scored_graph=scored_graph,
                ranked_edges=ranked_edges,
                validation_loader=validation_loader,
                metrics=metrics,
                top_k_values=top_k,
                random_repeats=config.random_repeats,
                seed=seed,
                skip_random_baselines=config.skip_random_baselines,
                checkpoint_path=eval_checkpoint_path,
            )
            evaluations.insert(0, "sample_size", int(sample_size))
            evaluations.insert(1, "seed", int(seed))
            save_csv(evaluations, eval_path, index=False)

            summary = {
                "experiment": "eap_ig_localization",
                "config": asdict(config),
                "sample_size": int(sample_size),
                "seed": int(seed),
                "paths": {
                    "project_root": str(project_root),
                    "output_root": str(output_root),
                    "output_dir": str(output_dir),
                    "edge_rankings": str(edge_path),
                    "node_rankings": str(node_path),
                    "topk_evaluations": str(eval_path),
                    "topk_evaluations_partial": str(eval_checkpoint_path),
                    "cumulative_mass": str(cumulative_path),
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
                    "target_raw_accuracy": prepared["target_raw_accuracy"],
                    "target_on_source_accuracy": prepared["target_on_source_accuracy"],
                    "named_entity_discovery_dir": prepared.get("named_entity_discovery_dir"),
                    "named_entity_discovery_summary": prepared.get("named_entity_discovery_summary"),
                },
                "graph_summary": {
                    "expanded_edge_count": int(scored_graph.real_edge_mask.sum().item()),
                    "ranked_edge_count": int(len(ranked_edges)),
                    "ranked_node_count": int(len(ranked_nodes)),
                    "resolved_top_k": [int(k) for k in top_k],
                },
                "concentration": concentration,
                "ranking_source": (
                    {
                        "kind": "imported_edge_rankings",
                        "edge_rankings_path": str(imported_ranking_path),
                        **(import_metadata or {}),
                    }
                    if import_metadata is not None
                    else {"kind": "localized_attribution"}
                ),
            }
            save_json(summary_path, summary)
            run_summaries.append(summary)

            del scored_graph
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    run_summaries = []
    concentration_rows = []
    run_edges = {}
    run_nodes = {}
    for summary_path in sorted(output_root.glob("sample_*/seed_*/localization_summary_sample_*_seed_*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sample_size = int(summary["sample_size"])
        seed = int(summary["seed"])
        run_summaries.append(summary)
        if "concentration" in summary:
            concentration_rows.append(summary["concentration"])
        edge_path = Path(summary["paths"]["edge_rankings"])
        node_path = Path(summary["paths"]["node_rankings"])
        if edge_path.is_file():
            run_edges[(sample_size, seed)] = pd.read_csv(edge_path)
        if node_path.is_file():
            run_nodes[(sample_size, seed)] = pd.read_csv(node_path)

    concentration_frame = pd.DataFrame(concentration_rows)
    concentration_path = output_root / "concentration_summary.csv"
    save_csv(concentration_frame, concentration_path, index=False)

    edge_stability_frame = edge_stability(run_edges, config.top_k)
    edge_stability_path = output_root / "edge_stability.csv"
    save_csv(edge_stability_frame, edge_stability_path, index=False)

    node_stability_frame = node_stability(run_nodes, config.top_k)
    node_stability_path = output_root / "node_stability.csv"
    save_csv(node_stability_frame, node_stability_path, index=False)

    all_evaluations = []
    for summary in run_summaries:
        eval_path = Path(summary["paths"]["topk_evaluations"])
        if eval_path.is_file():
            all_evaluations.append(pd.read_csv(eval_path))
    evaluation_frame = pd.concat(all_evaluations, ignore_index=True) if all_evaluations else pd.DataFrame()
    evaluation_path = output_root / "topk_evaluations.csv"
    save_csv(evaluation_frame, evaluation_path, index=False)

    manifest_path = output_root / f"localization_manifest_{timestamp_tag()}.json"
    manifest = {
        "experiment": "eap_ig_localization",
        "config": asdict(config),
        "paths": {
            "project_root": str(project_root),
            "output_root": str(output_root),
            "manifest": str(manifest_path),
            "concentration_summary": str(concentration_path),
            "edge_stability": str(edge_stability_path),
            "node_stability": str(node_stability_path),
            "topk_evaluations": str(evaluation_path),
        },
        "dataset_summary": {
            "dataset_mode": config.dataset_mode,
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "source_filter_model": prepared["dataset_filter_model_name"],
            "target_filtered_count": int(len(prepared["filtered_df"])),
            "target_raw_accuracy": prepared["target_raw_accuracy"],
            "target_on_source_accuracy": prepared["target_on_source_accuracy"],
            "named_entity_discovery_dir": prepared.get("named_entity_discovery_dir"),
            "named_entity_discovery_summary": prepared.get("named_entity_discovery_summary"),
        },
        "run_count": len(run_summaries),
        "runs": [
            {
                "sample_size": summary["sample_size"],
                "seed": summary["seed"],
                "summary": summary["paths"]["summary"],
            }
            for summary in run_summaries
        ],
    }
    save_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EAP-IG localization/distribution diagnostics for the animacy circuit."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset-filter-model", default="gpt2")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=list(DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K))
    parser.add_argument("--random-repeats", type=int, default=DEFAULT_RANDOM_REPEATS)
    parser.add_argument("--discovery-margin-threshold", type=float, default=DEFAULT_DISCOVERY_MARGIN_THRESHOLD)
    parser.add_argument("--filter-batch-size", type=int, default=50)
    parser.add_argument("--attribution-batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--budget-max-fraction", type=float, default=DEFAULT_EAP_BUDGET_MAX_FRACTION)
    parser.add_argument("--budget-floor", type=int, default=2000)
    parser.add_argument("--budget-tail-points", type=int, default=DEFAULT_EAP_BUDGET_TAIL_POINTS)
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
    parser.add_argument("--skip-random-baselines", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--import-edge-rankings-path", default=None)
    parser.add_argument("--import-summary-path", default=None)
    parser.add_argument("--import-sample-size", type=int, default=None)
    parser.add_argument("--import-seed", type=int, default=None)
    parser.add_argument(
        "--dataset-mode",
        choices=("semantic_filtered", "named_entity_truncated"),
        default="semantic_filtered",
    )
    parser.add_argument("--dataset-set", default=MODEL_SPECIFIC_CORRECT)
    parser.add_argument("--named-entity-discovery-dir", default=None)
    parser.add_argument(
        "--target-source",
        default="dataset/semantic_meaningful/named_entity_targets.json",
    )
    parser.add_argument(
        "--target-token-mode",
        choices=("first_token", "whole_entity_single_token"),
        default="first_token",
    )
    parser.add_argument("--start-path", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = LocalizationConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        sample_sizes=tuple(args.sample_sizes),
        seeds=tuple(args.seeds),
        top_k=tuple(args.top_k),
        random_repeats=args.random_repeats,
        discovery_margin_threshold=args.discovery_margin_threshold,
        filter_batch_size=args.filter_batch_size,
        attribution_batch_size=args.attribution_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        ig_steps=args.ig_steps,
        max_validation_examples=args.max_validation_examples,
        budget_max_fraction=args.budget_max_fraction,
        budget_floor=args.budget_floor,
        budget_tail_points=args.budget_tail_points,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        skip_random_baselines=args.skip_random_baselines,
        skip_existing=args.skip_existing,
        import_edge_rankings_path=args.import_edge_rankings_path,
        import_summary_path=args.import_summary_path,
        import_sample_size=args.import_sample_size,
        import_seed=args.import_seed,
        dataset_mode=args.dataset_mode,
        dataset_set=args.dataset_set,
        named_entity_discovery_dir=args.named_entity_discovery_dir,
        target_source=args.target_source,
        target_token_mode=args.target_token_mode,
    )
    manifest = run_localization_experiment(config, start=args.start_path)
    print(f"Saved EAP-IG localization outputs to {manifest['paths']['output_root']}")
    print(f"Manifest: {manifest['paths']['manifest']}")


if __name__ == "__main__":
    main()

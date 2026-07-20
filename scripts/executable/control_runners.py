from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from circuit_finder_core import (
    DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS,
    DEFAULT_DISCOVERY_MARGIN_THRESHOLD,
    add_concept_verb_positions,
    add_sequence_lengths,
    attribute_graph,
    build_budget_circuit,
    build_graph,
    canonical_model_name,
    collapsed_edge_groups,
    compute_sequence_metrics,
    concept_hook_name,
    date_tag,
    ensure_dir,
    find_metric_filtered_model_dataset_path,
    first_budget_reaching_faithfulness,
    induced_node_ranking,
    load_saved_ranked_edges,
    load_metric_filtered_model_success_dataset,
    load_model,
    load_model_context,
    make_dataloader,
    make_eap_accuracy_metric,
    normalize_concept_pair_metadata,
    pair_token_alignment_details,
    ranking_frame,
    resolve_animacy_circuit_root,
    resolve_eap_budget_grid,
    run_eap_budget_sweep,
    safe_model_name,
    sample_discovery_validation,
    save_csv,
    save_json,
    task_accuracy_summary,
    final_token_average_logit_difference,
    token_count_no_special,
    tokenizer_input_ids,
)


DEFAULT_VERB_NOISE_SIGMA_MULTIPLIERS = (
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
)
DEFAULT_CONTROL_AMBIGUITY_RELATIVE_MARGIN_THRESHOLD = 0.25
DEFAULT_CONTROL_AMBIGUITY_FAITHFULNESS_THRESHOLD = 0.75


@dataclass
class VerbNoiseControlConfig:
    model_name: str = "gpt2"
    dataset_filter_model_name: str = "gpt2"
    seed: int = 42
    filter_batch_size: int = 50
    evaluation_batch_size: int = 1
    target_filter_policy: str = "model_success"
    sigma: float = 0.0
    noise_site: str = "hook_resid_pre"
    output_day: str | None = None
    main_experiment_path: str | None = None
    max_budgets: int | None = None
    run_second_stage_discovery_on_ambiguous: bool = False
    ambiguity_relative_margin_threshold: float = DEFAULT_CONTROL_AMBIGUITY_RELATIVE_MARGIN_THRESHOLD
    ambiguity_faithfulness_threshold: float = DEFAULT_CONTROL_AMBIGUITY_FAITHFULNESS_THRESHOLD


@dataclass
class PrepositionControlConfig:
    model_name: str = "gpt2"
    dataset_filter_model_name: str = "gpt2"
    seed: int = 42
    filter_batch_size: int = 50
    evaluation_batch_size: int = 1
    target_filter_policy: str = "model_success"
    replacement_from: str = " by the"
    replacement_to: str = " near the"
    output_day: str | None = None
    main_experiment_path: str | None = None
    max_budgets: int | None = None
    run_second_stage_discovery_on_ambiguous: bool = False
    ambiguity_relative_margin_threshold: float = DEFAULT_CONTROL_AMBIGUITY_RELATIVE_MARGIN_THRESHOLD
    ambiguity_faithfulness_threshold: float = DEFAULT_CONTROL_AMBIGUITY_FAITHFULNESS_THRESHOLD


@dataclass
class BlimpPassivePrefixControlConfig:
    model_name: str = "gpt2"
    evaluation_batch_size: int = 32
    output_day: str | None = None
    main_experiment_path: str | None = None
    source_faithfulness_threshold: float = 0.85
    budget: int | None = None


def control_output_dir(
    project_root: Path,
    model_name: str,
    day: str | None,
    control_name: str,
) -> Path:
    resolved_day = date_tag() if day is None else day
    return ensure_dir(
        project_root
        / "results"
        / resolved_day
        / safe_model_name(canonical_model_name(model_name))
        / "controls"
        / control_name
    )


def _source_pair_key(row: pd.Series) -> str:
    uid = row.get("uid")
    if uid is not None and not pd.isna(uid):
        return str(uid)
    return hashlib.sha256(
        f"{row['clean_prefix']} || {row['corrupt_prefix']}".encode("utf-8")
    ).hexdigest()


def add_source_pair_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    if "source_pair_key" in keyed.columns and keyed["source_pair_key"].notna().all():
        keyed["source_pair_key"] = keyed["source_pair_key"].astype(str)
        return keyed
    keyed["source_pair_key"] = [
        _source_pair_key(row)
        for _, row in keyed.iterrows()
    ]
    return keyed


def load_control_prepared_inputs(
    *,
    project_root: Path,
    model_name: str,
    dataset_filter_model_name: str,
    target_filter_policy: str,
) -> dict[str, Any]:
    if target_filter_policy != "model_success":
        raise ValueError(
            "Control runners currently support only "
            "target_filter_policy='model_success' because they load the saved per-model filtered dataset."
        )

    resolved_model_name = canonical_model_name(model_name)
    metric_filtered_path = find_metric_filtered_model_dataset_path(
        project_root,
        resolved_model_name,
    )
    if metric_filtered_path is None:
        raise FileNotFoundError(
            f"No saved metric-filtered dataset found for {resolved_model_name}."
        )

    common_filter_model_names = tuple(DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS)
    print(
        f"Loading saved metric-filtered dataset for {resolved_model_name} "
        f"from {metric_filtered_path}"
    )
    filtered_df = load_metric_filtered_model_success_dataset(
        project_root=project_root,
        model_name=resolved_model_name,
        path=metric_filtered_path,
        common_filter_model_names=common_filter_model_names,
    )
    filtered_df = add_source_pair_keys(
        normalize_concept_pair_metadata(filtered_df)
    ).reset_index(drop=True)
    print(f"Loading target model {resolved_model_name}.")
    context = load_model_context(
        project_root,
        resolved_model_name,
        target_filter_model_names=common_filter_model_names,
    )
    return {
        **context,
        "requested_model_name": model_name,
        "dataset_filter_model_name": canonical_model_name(dataset_filter_model_name),
        "filtered_df": filtered_df,
        "target_raw_scored_df": filtered_df.copy(),
        "target_scored_df": filtered_df.copy(),
    }


def select_matching_rows(
    source_df: pd.DataFrame,
    selector_df: pd.DataFrame,
) -> pd.DataFrame:
    keyed_source = add_source_pair_keys(source_df)
    keyed_selector = add_source_pair_keys(selector_df)
    matched = keyed_source.merge(
        keyed_selector[["source_pair_key"]],
        on="source_pair_key",
        how="inner",
    )
    return matched.reset_index(drop=True).copy()


def _main_full_model_root(project_root: Path, model_name: str) -> Path:
    return project_root / "results" / "eap_ig" / safe_model_name(canonical_model_name(model_name))


def _latest_full_model_artifact(project_root: Path, model_name: str) -> Path | None:
    root = _main_full_model_root(project_root, model_name)
    if not root.is_dir():
        return None
    summary_candidates = sorted(
        root.glob("**/full_model/full_model_summary_*.json"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if summary_candidates:
        return summary_candidates[-1]
    full_model_dirs = sorted(
        {
            candidate.parent
            for candidate in root.glob("**/full_model/full_model_edges_*.csv")
        },
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if full_model_dirs:
        return full_model_dirs[-1]
    edge_candidates = sorted(
        root.glob("**/full_model/full_model_edges_*.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    return edge_candidates[-1] if edge_candidates else None


def _infer_summary_path_from_output_dir(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.glob("full_model_summary_*.json"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _coerce_existing_path(project_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_file() or path.is_dir():
        return path
    candidate = (project_root / path).resolve()
    if candidate.is_file() or candidate.is_dir():
        return candidate
    return path


def resolve_main_experiment_artifact(
    project_root: Path,
    model_name: str,
    main_artifact_or_rankings: str | Path | None,
) -> dict[str, Any]:
    target = _coerce_existing_path(project_root, main_artifact_or_rankings)
    if target is None:
        target = _latest_full_model_artifact(project_root, model_name)
    if target is None:
        raise FileNotFoundError(
            "No main full-model EAP artifact was provided and no saved full_model_summary_*.json was found."
        )

    summary_path: Path | None = None
    edge_path: Path | None = None
    node_path: Path | None = None
    budget_path: Path | None = None
    output_dir: Path

    if target.is_dir():
        output_dir = target
        summary_path = _infer_summary_path_from_output_dir(output_dir)
    else:
        output_dir = target.parent
        if target.name.startswith("full_model_summary_") and target.suffix == ".json":
            summary_path = target
        elif target.name.startswith("full_model_edges_") and target.suffix == ".csv":
            edge_path = target
            summary_path = _infer_summary_path_from_output_dir(output_dir)
        elif target.name.startswith("full_model_budget_sweep_") and target.suffix == ".csv":
            budget_path = target
            summary_path = _infer_summary_path_from_output_dir(output_dir)
        else:
            raise ValueError(
                f"Unsupported main_experiment_path {target}. "
                "Pass a full_model_summary_*.json, full_model_edges_*.csv, full_model_budget_sweep_*.csv, or a full_model directory."
            )

    summary: dict[str, Any] = {}
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        paths = summary.get("paths", {})
        if edge_path is None and paths.get("edge_rankings"):
            edge_path = Path(paths["edge_rankings"])
        if node_path is None and paths.get("node_rankings"):
            node_path = Path(paths["node_rankings"])
        if budget_path is None and paths.get("budget_sweep"):
            budget_path = Path(paths["budget_sweep"])

    if edge_path is None:
        edges = sorted(
            output_dir.glob("full_model_edges_*.csv"),
            key=lambda candidate: candidate.stat().st_mtime,
        )
        edge_path = edges[-1] if edges else None
    if node_path is None:
        nodes = sorted(
            output_dir.glob("full_model_nodes_*.csv"),
            key=lambda candidate: candidate.stat().st_mtime,
        )
        node_path = nodes[-1] if nodes else None
    if budget_path is None:
        budgets = sorted(
            output_dir.glob("full_model_budget_sweep_*.csv"),
            key=lambda candidate: candidate.stat().st_mtime,
        )
        budget_path = budgets[-1] if budgets else None

    if edge_path is None or not edge_path.is_file():
        raise FileNotFoundError(f"Could not resolve full-model edge rankings under {output_dir}")
    loaded = load_saved_ranked_edges(edge_path, node_path)
    if loaded is None:
        raise ValueError(f"Could not parse ranked edges from {edge_path}")
    ranked_edges, edge_frame, node_frame = loaded

    budget_frame = pd.read_csv(budget_path) if budget_path is not None and budget_path.is_file() else pd.DataFrame()
    if not budget_frame.empty and "collapsed_edge_budget" in budget_frame.columns:
        budget_frame = (
            budget_frame.copy()
            .assign(collapsed_edge_budget=lambda frame: frame["collapsed_edge_budget"].astype(int))
            .sort_values("collapsed_edge_budget")
            .drop_duplicates(subset=["collapsed_edge_budget"], keep="last")
            .reset_index(drop=True)
        )

    return {
        "summary": summary,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "output_dir": str(output_dir),
        "edge_path": str(edge_path),
        "node_path": str(node_path) if node_path is not None else None,
        "budget_path": str(budget_path) if budget_path is not None else None,
        "edge_frame": edge_frame,
        "node_frame": node_frame,
        "budget_frame": budget_frame,
        "ranked_edges": ranked_edges,
    }


def _validate_blimp_prefix_flags(df: pd.DataFrame) -> None:
    if "one_prefix_method" in df.columns and not df["one_prefix_method"].fillna(False).all():
        raise ValueError("BLiMP passive prefix control requires one_prefix_method=true for every row.")
    if "simple_LM_method" in df.columns and not df["simple_LM_method"].fillna(False).all():
        raise ValueError("BLiMP passive prefix control requires simple_LM_method=true for every row.")


def load_local_blimp_prefix_dataset(
    project_root: Path,
    blimp_config: str,
) -> tuple[pd.DataFrame, Path]:
    dataset_path = project_root / "dataset" / "blimp" / f"{blimp_config}.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Expected local BLiMP prefix dataset at {dataset_path}"
        )
    return pd.read_json(dataset_path, lines=True), dataset_path


def prepare_blimp_passive_prefix_rows(
    raw_df: pd.DataFrame,
    tokenizer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_blimp_prefix_flags(raw_df)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, row in raw_df.reset_index(drop=True).iterrows():
        prefix = str(row["one_prefix_prefix"]).strip()
        if not prefix:
            failures.append(
                {
                    "row": int(idx),
                    "pairID": row.get("pairID"),
                    "UID": row.get("UID"),
                    "prefix": prefix,
                    "sentence_good": row.get("sentence_good"),
                    "sentence_bad": row.get("sentence_bad"),
                    "failure_reason": "empty_prefix",
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
                "prefix": prefix,
                "seq_len": int(token_count_no_special(tokenizer, prefix)),
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
            "prefix",
            "sentence_good",
            "sentence_bad",
            "failure_reason",
        ],
    )
    return prepared, failure_frame


def resolve_main_experiment_settings(
    config: VerbNoiseControlConfig | PrepositionControlConfig,
    prepared: dict[str, Any],
    main_artifact: dict[str, Any],
) -> dict[str, Any]:
    summary = main_artifact["summary"]
    summary_config = summary.get("config", {})
    dataset_summary = summary.get("dataset_summary", {})

    sample_size = int(summary_config.get("discovery_sample_size", dataset_summary.get("discovery_count", 500)))
    split_seed = int(summary_config.get("seed", config.seed))
    margin_threshold = summary_config.get("discovery_margin_threshold", DEFAULT_DISCOVERY_MARGIN_THRESHOLD)
    if margin_threshold is not None:
        margin_threshold = float(margin_threshold)

    if dataset_summary.get("target_filter_policy", config.target_filter_policy) != config.target_filter_policy:
        raise ValueError(
            "Main experiment target_filter_policy does not match control config: "
            f"{dataset_summary.get('target_filter_policy')!r} != {config.target_filter_policy!r}"
        )
    if canonical_model_name(dataset_summary.get("target_model", config.model_name)) != canonical_model_name(config.model_name):
        raise ValueError(
            "Main experiment target_model does not match control config: "
            f"{dataset_summary.get('target_model')!r} != {canonical_model_name(config.model_name)!r}"
        )
    if canonical_model_name(
        dataset_summary.get("source_filter_model", config.dataset_filter_model_name)
    ) != canonical_model_name(config.dataset_filter_model_name):
        raise ValueError(
            "Main experiment source_filter_model does not match control config: "
            f"{dataset_summary.get('source_filter_model')!r} != {canonical_model_name(config.dataset_filter_model_name)!r}"
        )
    expected_count = dataset_summary.get("target_filtered_count")
    if expected_count is not None and int(expected_count) != int(len(prepared["filtered_df"])):
        raise ValueError(
            "Main experiment target_filtered_count does not match the current filtered dataset: "
            f"{expected_count} != {len(prepared['filtered_df'])}"
        )

    discovery_df, validation_df, sample_signature = sample_discovery_validation(
        prepared["filtered_df"],
        discovery_sample_size=sample_size,
        seed=split_seed,
        discovery_margin_threshold=margin_threshold,
    )
    expected_signature = dataset_summary.get("discovery_sample_signature")
    if expected_signature is not None and expected_signature != sample_signature:
        raise ValueError(
            "Current filtered dataset does not reproduce the discovery sample from the main experiment."
        )

    return {
        "discovery_df": discovery_df,
        "validation_df": validation_df,
        "sample_signature": sample_signature,
        "discovery_sample_size": sample_size,
        "split_seed": split_seed,
        "discovery_margin_threshold": margin_threshold,
        "attribution_batch_size": int(summary_config.get("attribution_batch_size", 8)),
        "evaluation_batch_size": int(config.evaluation_batch_size),
        "ig_steps": int(summary_config.get("ig_steps", 5)),
    }


def resolve_main_budgets(
    main_artifact: dict[str, Any],
    ranked_edge_count: int,
    max_budgets: int | None = None,
) -> list[int]:
    budget_frame = main_artifact["budget_frame"]
    if not budget_frame.empty and "collapsed_edge_budget" in budget_frame.columns:
        budgets = [
            int(value)
            for value in budget_frame["collapsed_edge_budget"].tolist()
            if int(value) <= ranked_edge_count
        ]
        return budgets[:max_budgets] if max_budgets is not None else budgets

    summary = main_artifact["summary"]
    graph_summary = summary.get("graph_summary", {})
    resolved_budget_grid = graph_summary.get("resolved_budget_grid")
    if resolved_budget_grid:
        budgets = [int(value) for value in resolved_budget_grid if int(value) <= ranked_edge_count]
        return budgets[:max_budgets] if max_budgets is not None else budgets

    config = summary.get("config", {})
    budgets = resolve_eap_budget_grid(
        ranked_edge_count,
        budgets=tuple(int(value) for value in config.get("budgets", []) or []) or None,
        budget_max_fraction=float(config.get("budget_max_fraction", 0.15)),
        budget_floor=int(config.get("budget_floor", 2000)),
        budget_tail_points=int(config.get("budget_tail_points", 20)),
    )
    return budgets[:max_budgets] if max_budgets is not None else budgets


def _iter_batch_frames(df: pd.DataFrame, batch_size: int):
    for _, group in df.groupby("seq_len", sort=True):
        for start in range(0, len(group), batch_size):
            yield group.iloc[start : start + batch_size].reset_index(drop=True).copy()


def _zero_forward_node_hooks(graph) -> list[tuple[str, Callable[..., torch.Tensor]]]:
    hook_names = sorted(
        {
            node.out_hook
            for node in graph.nodes.values()
            if getattr(node, "out_hook", None)
        }
    )

    def zero_hook(activations: torch.Tensor, hook):
        del hook
        return torch.zeros_like(activations)

    return [(hook_name, zero_hook) for hook_name in hook_names]


def _retained_circuit_logits(
    model,
    graph,
    texts: Sequence[str],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from einops import einsum
    from eap.utils import make_hooks_and_matrices, tokenize_plus

    assert model.cfg.use_attn_result, (
        "Model must be configured to use attention result (model.cfg.use_attn_result)"
    )
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, (
            "Model must be configured to ungroup grouped attention "
            "(model.cfg.ungroup_grouped_attention)"
        )

    graph.prune()
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(
            in_graph_matrix,
            node_fully_in_graph,
            "forward backward, forward -> forward backward",
        )
    else:
        neuron_matrix = None

    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix

    zero_hooks = _zero_forward_node_hooks(graph)
    clean_final_logits_parts: list[torch.Tensor] = []
    retained_final_logits_parts: list[torch.Tensor] = []
    frame = pd.DataFrame({"prefix": list(texts)})
    frame["seq_len"] = [int(token_count_no_special(model.tokenizer, text)) for text in frame["prefix"]]
    for batch_df in tqdm(
        list(_iter_batch_frames(frame, batch_size)),
        desc="Retained-circuit prefix eval",
        leave=False,
    ):
        batch_texts = batch_df["prefix"].tolist()
        clean_tokens, attention_mask, _, n_pos = tokenize_plus(model, batch_texts)
        batch_size_value = len(batch_texts)

        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(
            model,
            graph,
            batch_size_value,
            n_pos,
            None,
        )
        input_construction_hooks = _make_input_construction_hooks(
            model=model,
            graph=graph,
            activation_difference=activation_difference,
            in_graph_matrix=in_graph_matrix,
            neuron_matrix=neuron_matrix,
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=zero_hooks + fwd_hooks_corrupted):
                _ = model(clean_tokens, attention_mask=attention_mask)
            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)
            with model.hooks(fwd_hooks=fwd_hooks_clean + input_construction_hooks):
                retained_logits = model(clean_tokens, attention_mask=attention_mask)

        clean_final_logits_parts.append(clean_logits[:, -1, :].detach().cpu())
        retained_final_logits_parts.append(retained_logits[:, -1, :].detach().cpu())

    return (
        torch.cat(clean_final_logits_parts, dim=0),
        torch.cat(retained_final_logits_parts, dim=0),
    )


def evaluate_blimp_passive_prefix_control(
    *,
    model,
    graph,
    df: pd.DataFrame,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    batch_size: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "pairID",
                "UID",
                "field",
                "linguistics_term",
                "sentence_good",
                "sentence_bad",
                "prefix",
                "full_animate_logit_mean",
                "full_inanimate_logit_mean",
                "circuit_animate_logit_mean",
                "circuit_inanimate_logit_mean",
                "full_logit_diff",
                "circuit_logit_diff",
                "logit_diff_delta_circuit_minus_full",
                "full_prefers_animate",
                "circuit_prefers_animate",
                "flip_to_animate",
                "flip_away_from_animate",
            ]
        )

    clean_logits, retained_logits = _retained_circuit_logits(
        model=model,
        graph=graph,
        texts=df["prefix"].astype(str).tolist(),
        batch_size=batch_size,
    )
    animate_ids = animate_ids_tensor.detach().to(device=clean_logits.device, dtype=torch.long)
    inanimate_ids = inanimate_ids_tensor.detach().to(device=clean_logits.device, dtype=torch.long)

    full_animate = clean_logits[:, animate_ids].mean(dim=-1).numpy()
    full_inanimate = clean_logits[:, inanimate_ids].mean(dim=-1).numpy()
    circuit_animate = retained_logits[:, animate_ids].mean(dim=-1).numpy()
    circuit_inanimate = retained_logits[:, inanimate_ids].mean(dim=-1).numpy()
    full_logit_diff = (clean_logits[:, animate_ids].mean(dim=-1) - clean_logits[:, inanimate_ids].mean(dim=-1)).numpy()
    circuit_logit_diff = (
        retained_logits[:, animate_ids].mean(dim=-1)
        - retained_logits[:, inanimate_ids].mean(dim=-1)
    ).numpy()

    result = df.copy()
    result["full_animate_logit_mean"] = full_animate
    result["full_inanimate_logit_mean"] = full_inanimate
    result["circuit_animate_logit_mean"] = circuit_animate
    result["circuit_inanimate_logit_mean"] = circuit_inanimate
    result["full_logit_diff"] = full_logit_diff
    result["circuit_logit_diff"] = circuit_logit_diff
    result["logit_diff_delta_circuit_minus_full"] = (
        result["circuit_logit_diff"] - result["full_logit_diff"]
    )
    result["full_prefers_animate"] = result["full_logit_diff"] > 0
    result["circuit_prefers_animate"] = result["circuit_logit_diff"] > 0
    result["flip_to_animate"] = (~result["full_prefers_animate"]) & result["circuit_prefers_animate"]
    result["flip_away_from_animate"] = (
        result["full_prefers_animate"] & (~result["circuit_prefers_animate"])
    )
    return result.reset_index(drop=True)


def summarize_blimp_passive_prefix_control(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "example_count": 0,
            "full_model_accuracy": 0.0,
            "circuit_accuracy": 0.0,
            "accuracy_delta_circuit_minus_full": 0.0,
            "full_model_logit_diff_mean": 0.0,
            "circuit_logit_diff_mean": 0.0,
            "logit_diff_delta_mean": 0.0,
            "logit_diff_delta_std": 0.0,
            "circuit_fix_count": 0,
            "circuit_break_count": 0,
            "circuit_fix_rate": 0.0,
            "circuit_break_rate": 0.0,
        }

    return {
        "example_count": int(len(rows)),
        "full_model_accuracy": float(rows["full_prefers_animate"].mean()),
        "circuit_accuracy": float(rows["circuit_prefers_animate"].mean()),
        "accuracy_delta_circuit_minus_full": float(
            rows["circuit_prefers_animate"].mean() - rows["full_prefers_animate"].mean()
        ),
        "full_model_logit_diff_mean": float(rows["full_logit_diff"].mean()),
        "circuit_logit_diff_mean": float(rows["circuit_logit_diff"].mean()),
        "logit_diff_delta_mean": float(rows["logit_diff_delta_circuit_minus_full"].mean()),
        "logit_diff_delta_std": (
            float(rows["logit_diff_delta_circuit_minus_full"].std(ddof=0))
            if len(rows) > 1
            else 0.0
        ),
        "circuit_fix_count": int(rows["flip_to_animate"].sum()),
        "circuit_break_count": int(rows["flip_away_from_animate"].sum()),
        "circuit_fix_rate": float(rows["flip_to_animate"].mean()),
        "circuit_break_rate": float(rows["flip_away_from_animate"].mean()),
    }


def _pair_seed_for_row(row: pd.Series, base_seed: int) -> int:
    key = f"{base_seed}|{row['source_pair_key']}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def add_control_pair_seeds(df: pd.DataFrame, base_seed: int) -> pd.DataFrame:
    seeded = add_source_pair_keys(df)
    seeded["control_pair_seed"] = [
        _pair_seed_for_row(row, base_seed)
        for _, row in seeded.iterrows()
    ]
    return seeded


def require_exact_split_match(
    control_df: pd.DataFrame,
    source_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    matched = select_matching_rows(control_df, source_df)
    source_keys = add_source_pair_keys(source_df)["source_pair_key"].astype(str).tolist()
    matched_keys = matched["source_pair_key"].astype(str).tolist()
    if len(matched) != len(source_df) or set(matched_keys) != set(source_keys):
        missing = sorted(set(source_keys) - set(matched_keys))
        raise ValueError(
            f"{split_name} rows were not preserved exactly in the control dataset. "
            f"Expected {len(source_df)} rows, matched {len(matched)}. "
            f"First missing keys: {missing[:5]}"
        )
    return matched.reset_index(drop=True).copy()


def _positioned_vector_hook(
    positions: list[int],
    vectors: torch.Tensor,
):
    def hook_fn(activations: torch.Tensor, hook):
        updated = activations.clone()
        batch_indices = torch.arange(updated.shape[0], device=updated.device)
        position_tensor = torch.tensor(positions, device=updated.device, dtype=torch.long)
        updated[batch_indices, position_tensor, :] = (
            updated[batch_indices, position_tensor, :]
            + vectors.to(device=updated.device, dtype=updated.dtype)
        )
        return updated

    return hook_fn


def _sample_noise_vectors(
    batch_df: pd.DataFrame,
    d_model: int,
    sigma: float,
    device: torch.device | str,
) -> torch.Tensor:
    if sigma == 0.0:
        return torch.zeros((len(batch_df), d_model), device=device)
    rows: list[torch.Tensor] = []
    for pair_seed in batch_df["control_pair_seed"].astype(int).tolist():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(pair_seed))
        rows.append(torch.randn(d_model, generator=generator, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device=device) * float(sigma)


def make_batch_noise_hooks(
    batch_df: pd.DataFrame,
    hook_name: str,
    d_model: int,
    sigma: float,
    device: torch.device | str,
) -> tuple[list[tuple[str, Callable[..., torch.Tensor]]], list[tuple[str, Callable[..., torch.Tensor]]]]:
    vectors = _sample_noise_vectors(batch_df, d_model, sigma, device=device)
    clean_positions = batch_df["clean_verb_token_position"].astype(int).tolist()
    corrupt_positions = batch_df["corrupt_verb_token_position"].astype(int).tolist()
    return (
        [(hook_name, _positioned_vector_hook(clean_positions, vectors))],
        [(hook_name, _positioned_vector_hook(corrupt_positions, vectors))],
    )


def prepare_verb_noise_control_rows(
    df: pd.DataFrame,
    tokenizer,
) -> pd.DataFrame:
    keyed = add_source_pair_keys(df)
    validated = add_concept_verb_positions(keyed, tokenizer)
    if len(validated) != len(df):
        raise ValueError(
            "Verb-noise control requires every selected row to retain aligned single-token verbs."
        )
    return validated.reset_index(drop=True).copy()


def prepare_verb_noise_control_dataframe(
    df: pd.DataFrame,
    model,
    tokenizer,
    seed: int,
) -> pd.DataFrame:
    prepared = prepare_verb_noise_control_rows(df, tokenizer)
    with_lengths = add_sequence_lengths(prepared, model)
    if len(with_lengths) != len(prepared):
        raise ValueError(
            "Verb-noise control requires clean/corrupt token-count alignment after BOS tokenization."
        )
    return add_control_pair_seeds(with_lengths, seed).reset_index(drop=True).copy()


def make_control_eap_normalized_recovery_metric(
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    eps: float = 1e-6,
):
    def metric(
        logits: torch.Tensor,
        clean_logits: torch.Tensor | None,
        input_lengths: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        del input_lengths
        if clean_logits is None:
            raise ValueError("clean_logits is required to compute normalized recovery.")

        patched_logit_diff = final_token_average_logit_difference(
            logits,
            animate_ids_tensor,
            inanimate_ids_tensor,
        )
        clean_logit_diff = final_token_average_logit_difference(
            clean_logits,
            animate_ids_tensor,
            inanimate_ids_tensor,
        )
        corrupt_logit_diff = label.to(
            device=logits.device,
            dtype=clean_logit_diff.dtype,
        )
        denominator = clean_logit_diff - corrupt_logit_diff
        if not torch.all(torch.abs(denominator) > eps):
            raise ValueError(
                "Each sample must have a sufficiently non-zero clean-corrupt logit-difference margin."
            )
        recovery = (patched_logit_diff - corrupt_logit_diff) / denominator
        return recovery.mean()

    return metric


def make_control_eap_normalized_recovery_vector_metric(
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    eps: float = 1e-6,
):
    def metric(
        logits: torch.Tensor,
        clean_logits: torch.Tensor | None,
        input_lengths: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        del input_lengths
        if clean_logits is None:
            raise ValueError("clean_logits is required to compute normalized recovery.")

        patched_logit_diff = final_token_average_logit_difference(
            logits,
            animate_ids_tensor,
            inanimate_ids_tensor,
        )
        clean_logit_diff = final_token_average_logit_difference(
            clean_logits,
            animate_ids_tensor,
            inanimate_ids_tensor,
        )
        corrupt_logit_diff = label.to(
            device=logits.device,
            dtype=clean_logit_diff.dtype,
        )
        denominator = clean_logit_diff - corrupt_logit_diff
        if not torch.all(torch.abs(denominator) > eps):
            raise ValueError(
                "Each sample must have a sufficiently non-zero clean-corrupt logit-difference margin."
            )
        return (patched_logit_diff - corrupt_logit_diff) / denominator

    return metric


def make_control_eap_metrics(
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
) -> dict[str, Callable[..., torch.Tensor]]:
    return {
        "attribute": make_control_eap_normalized_recovery_metric(
            animate_ids_tensor,
            inanimate_ids_tensor,
        ),
        "faithfulness": make_control_eap_normalized_recovery_vector_metric(
            animate_ids_tensor,
            inanimate_ids_tensor,
        ),
        "accuracy": make_eap_accuracy_metric(
            animate_ids_tensor,
            inanimate_ids_tensor,
        ),
    }


def estimate_verb_noise_activation_scale(
    df: pd.DataFrame,
    model,
    batch_size: int,
    hook_name: str,
) -> float:
    from eap.utils import tokenize_plus

    values: list[torch.Tensor] = []
    model_device = next(model.parameters()).device

    for batch_df in tqdm(
        list(_iter_batch_frames(df, batch_size)),
        desc="Estimating verb activation scale",
        leave=False,
    ):
        clean_texts = batch_df["clean_prefix"].tolist()
        corrupt_texts = batch_df["corrupt_prefix"].tolist()
        clean_tokens, attention_mask, _, _ = tokenize_plus(model, clean_texts)
        corrupt_tokens, _, _, _ = tokenize_plus(model, corrupt_texts)

        clean_positions = torch.tensor(
            batch_df["clean_verb_token_position"].astype(int).tolist(),
            device=model_device,
            dtype=torch.long,
        )
        corrupt_positions = torch.tensor(
            batch_df["corrupt_verb_token_position"].astype(int).tolist(),
            device=model_device,
            dtype=torch.long,
        )
        clean_store: list[torch.Tensor] = []
        corrupt_store: list[torch.Tensor] = []

        def clean_hook_fn(activations: torch.Tensor, hook):
            batch_idx = torch.arange(activations.shape[0], device=activations.device)
            clean_store.append(
                activations[batch_idx, clean_positions, :].detach().cpu()
            )
            return activations

        def corrupt_hook_fn(activations: torch.Tensor, hook):
            batch_idx = torch.arange(activations.shape[0], device=activations.device)
            corrupt_store.append(
                activations[batch_idx, corrupt_positions, :].detach().cpu()
            )
            return activations

        with torch.inference_mode():
            _ = model.run_with_hooks(
                clean_tokens,
                attention_mask=attention_mask,
                fwd_hooks=[(hook_name, clean_hook_fn)],
            )
            _ = model.run_with_hooks(
                corrupt_tokens,
                attention_mask=attention_mask,
                fwd_hooks=[(hook_name, corrupt_hook_fn)],
            )

        values.extend(clean_store)
        values.extend(corrupt_store)

    if not values:
        raise ValueError("Could not collect verb activations for sigma calibration.")

    activations = torch.cat(values, dim=0).to(dtype=torch.float32)
    return float(torch.sqrt(torch.mean(torch.square(activations))).item())


def compute_noisy_sequence_metrics(
    df: pd.DataFrame,
    model,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    batch_size: int,
    sigma: float,
    hook_name: str,
) -> pd.DataFrame:
    from eap.utils import tokenize_plus

    clean_metric_map: dict[str, float] = {}
    corrupt_metric_map: dict[str, float] = {}
    animate_ids = animate_ids_tensor.to(model.cfg.device)
    inanimate_ids = inanimate_ids_tensor.to(model.cfg.device)

    for batch_df in tqdm(
        list(_iter_batch_frames(df, batch_size)),
        desc=f"Scoring noisy control sigma={sigma:g}",
        leave=False,
    ):
        clean_texts = batch_df["clean_prefix"].tolist()
        corrupt_texts = batch_df["corrupt_prefix"].tolist()
        clean_tokens, attention_mask, _, _ = tokenize_plus(model, clean_texts)
        corrupt_tokens, _, _, _ = tokenize_plus(model, corrupt_texts)

        clean_noise_hooks, corrupt_noise_hooks = make_batch_noise_hooks(
            batch_df=batch_df,
            hook_name=hook_name,
            d_model=model.cfg.d_model,
            sigma=sigma,
            device=model.cfg.device,
        )

        with torch.inference_mode():
            clean_logits = model.run_with_hooks(
                clean_tokens,
                attention_mask=attention_mask,
                fwd_hooks=clean_noise_hooks,
            )
            corrupt_logits = model.run_with_hooks(
                corrupt_tokens,
                attention_mask=attention_mask,
                fwd_hooks=corrupt_noise_hooks,
            )

        clean_metric = (
            clean_logits[:, -1, animate_ids].mean(dim=-1)
            - clean_logits[:, -1, inanimate_ids].mean(dim=-1)
        )
        corrupt_metric = (
            corrupt_logits[:, -1, animate_ids].mean(dim=-1)
            - corrupt_logits[:, -1, inanimate_ids].mean(dim=-1)
        )

        for key, clean_value, corrupt_value in zip(
            batch_df["source_pair_key"].tolist(),
            clean_metric.detach().cpu().tolist(),
            corrupt_metric.detach().cpu().tolist(),
        ):
            clean_metric_map[str(key)] = float(clean_value)
            corrupt_metric_map[str(key)] = float(corrupt_value)

    scored = df.copy()
    scored["clean_metric"] = [clean_metric_map[str(key)] for key in scored["source_pair_key"]]
    scored["corrupt_metric"] = [corrupt_metric_map[str(key)] for key in scored["source_pair_key"]]
    return scored


def sweep_verb_noise_sigmas(
    df: pd.DataFrame,
    model,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    batch_size: int,
    sigma_multipliers: Sequence[float] = DEFAULT_VERB_NOISE_SIGMA_MULTIPLIERS,
    hook_name: str = "blocks.0.hook_resid_pre",
) -> dict[str, Any]:
    activation_scale = estimate_verb_noise_activation_scale(
        df=df,
        model=model,
        batch_size=batch_size,
        hook_name=hook_name,
    )
    original_summary = task_accuracy_summary(df)
    rows: list[dict[str, Any]] = []
    for multiplier in sigma_multipliers:
        sigma = float(multiplier) * activation_scale
        scored = compute_noisy_sequence_metrics(
            df=df,
            model=model,
            animate_ids_tensor=animate_ids_tensor,
            inanimate_ids_tensor=inanimate_ids_tensor,
            batch_size=batch_size,
            sigma=sigma,
            hook_name=hook_name,
        )
        margin = scored["clean_metric"] - scored["corrupt_metric"]
        margin_mean = float(margin.mean())
        rows.append(
            {
                "sigma_multiplier": float(multiplier),
                "sigma": float(sigma),
                "clean_metric_mean": float(scored["clean_metric"].mean()),
                "corrupt_metric_mean": float(scored["corrupt_metric"].mean()),
                "margin_mean": margin_mean,
                "absolute_mean_margin": abs(margin_mean),
                "mean_absolute_margin": float(margin.abs().mean()),
            }
        )
    sweep_df = pd.DataFrame(rows).sort_values(["absolute_mean_margin", "sigma"]).reset_index(drop=True)
    return {
        "activation_scale": float(activation_scale),
        "original_summary": original_summary,
        "sweep_df": sweep_df,
    }


def select_verb_noise_sigma(
    sweep_df: pd.DataFrame,
    tolerance: float | None = None,
) -> dict[str, Any]:
    if sweep_df.empty:
        raise ValueError("Sigma sweep is empty.")
    selectable = sweep_df.copy()
    if "absolute_mean_margin" not in selectable.columns:
        selectable["absolute_mean_margin"] = selectable["margin_mean"].abs()
    ordered = selectable.sort_values(["absolute_mean_margin", "sigma"]).reset_index(drop=True)
    best_abs_margin = float(ordered.loc[0, "absolute_mean_margin"])
    tolerance_value = max(1e-3, 0.05 * best_abs_margin) if tolerance is None else float(tolerance)
    tied = ordered[ordered["absolute_mean_margin"] <= (best_abs_margin + tolerance_value)].copy()
    chosen = tied.sort_values("sigma", ascending=True).iloc[0].to_dict()
    chosen["tie_tolerance"] = float(tolerance_value)
    chosen["warning_overcorruption"] = float(chosen["margin_mean"]) < 0.0
    return chosen


def export_selected_sigma(
    output_path: Path,
    selected: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = dict(selected)
    if extra:
        payload.update(extra)
    save_json(output_path, payload)
    return output_path


def build_preposition_control_dataframe(
    df: pd.DataFrame,
    tokenizer,
    replacement_from: str = " by the",
    replacement_to: str = " near the",
) -> pd.DataFrame:
    if not replacement_to.startswith(" "):
        raise ValueError("replacement_to must begin with a space to preserve token boundaries.")
    replacement_token = replacement_to.split()[0]
    if len(tokenizer_input_ids(tokenizer, f" {replacement_token}")) != 1:
        raise ValueError(
            f"{replacement_to!r} is not usable for this model because {replacement_token!r} is not a single token."
        )

    rewritten = add_source_pair_keys(normalize_concept_pair_metadata(df))
    if not rewritten["clean_prefix"].astype(str).str.endswith(replacement_from).all():
        raise ValueError(f"Not every clean prefix ends with {replacement_from!r}.")
    if not rewritten["corrupt_prefix"].astype(str).str.endswith(replacement_from).all():
        raise ValueError(f"Not every corrupt prefix ends with {replacement_from!r}.")

    rewritten["clean_prefix"] = rewritten["clean_prefix"].astype(str).str.replace(
        f"{replacement_from}$",
        replacement_to,
        regex=True,
    )
    rewritten["corrupt_prefix"] = rewritten["corrupt_prefix"].astype(str).str.replace(
        f"{replacement_from}$",
        replacement_to,
        regex=True,
    )
    rewritten["control_type"] = "by_to_near"

    invalid_rows: list[dict[str, Any]] = []
    metadata_available = {"patient", "clean_verb", "corrupt_verb"}.issubset(rewritten.columns)
    for idx, row in rewritten.iterrows():
        details = pair_token_alignment_details(row, tokenizer, metadata_available=metadata_available)
        if not details["pair_ok"]:
            invalid_rows.append(
                {
                    "row": int(idx),
                    "clean_prefix": row["clean_prefix"],
                    "corrupt_prefix": row["corrupt_prefix"],
                    "clean_len": details["clean_len"],
                    "corrupt_len": details["corrupt_len"],
                }
            )
            continue
        if token_count_no_special(tokenizer, row["clean_prefix"]) != token_count_no_special(tokenizer, row["corrupt_prefix"]):
            invalid_rows.append(
                {
                    "row": int(idx),
                    "clean_prefix": row["clean_prefix"],
                    "corrupt_prefix": row["corrupt_prefix"],
                    "clean_len": token_count_no_special(tokenizer, row["clean_prefix"]),
                    "corrupt_len": token_count_no_special(tokenizer, row["corrupt_prefix"]),
                }
            )
    if invalid_rows:
        raise ValueError(
            "The by->near control does not preserve model-token alignment for every pair. "
            f"First failures: {invalid_rows[:5]}"
        )
    return rewritten.reset_index(drop=True).copy()


def _make_input_construction_hooks(
    *,
    model,
    graph,
    activation_difference: torch.Tensor,
    in_graph_matrix: torch.Tensor,
    neuron_matrix: torch.Tensor | None,
):
    from eap.graph import AttentionNode
    from einops import einsum

    if model.cfg.use_normalization_before_and_after:
        attention_head_mask = torch.zeros(
            (graph.n_forward, model.cfg.n_layers),
            device=model.cfg.device,
            dtype=model.cfg.dtype,
        )
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_mask):
        def input_construction_hook(activations, hook):
            if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]
                clean_attention_results = einsum(
                    activation_matrix[1, :, :, : len(in_graph_vector)],
                    attention_head_mask[: len(in_graph_vector)],
                    "batch pos previous hidden, previous layer -> batch pos layer hidden",
                )

                if neuron_mask is not None:
                    non_attention_update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_mask[: len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden",
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_mask[: len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden",
                    )
                else:
                    non_attention_update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous ..., previous -> batch pos ... hidden",
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden",
                    )

                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = clean_attention_results.unsqueeze(2) + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    corrupted_attention_results = clean_attention_results + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)

                update = non_attention_update
                valid_layers = attention_head_mask[: len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])
            else:
                activation_differences = activation_matrix
                if neuron_mask is not None:
                    update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_mask[: len(in_graph_vector)],
                        in_graph_vector,
                        "batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden",
                    )
                else:
                    update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        "batch pos previous hidden, previous ... -> batch pos ... hidden",
                    )
            activations = activations.clone()
            activations += update
            return activations

        return input_construction_hook

    hooks: list[tuple[str, Callable[..., torch.Tensor]]] = []
    for layer in range(model.cfg.n_layers):
        if any(graph.nodes[f"a{layer}.h{head}"].in_graph for head in range(model.cfg.n_heads)) and not (
            neuron_matrix is None
            and all(
                parent_edge.in_graph
                for head in range(model.cfg.n_heads)
                for parent_edge in graph.nodes[f"a{layer}.h{head}"].parent_edges
            )
        ):
            for i, letter in enumerate("qkv"):
                node = graph.nodes[f"a{layer}.h0"]
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                hooks.append(
                    (
                        node.qkv_inputs[i],
                        make_input_construction_hook(
                            activation_difference,
                            in_graph_matrix[:prev_index, bwd_index],
                            neuron_matrix,
                        ),
                    )
                )

        if graph.nodes[f"m{layer}"].in_graph and not (
            neuron_matrix is None
            and all(parent_edge.in_graph for parent_edge in graph.nodes[f"m{layer}"].parent_edges)
        ):
            node = graph.nodes[f"m{layer}"]
            prev_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            hooks.append(
                (
                    node.in_hook,
                    make_input_construction_hook(
                        activation_difference,
                        in_graph_matrix[:prev_index, bwd_index],
                        neuron_matrix,
                    ),
                )
            )

    if not (
        neuron_matrix is None
        and all(parent_edge.in_graph for parent_edge in graph.nodes["logits"].parent_edges)
    ):
        node = graph.nodes["logits"]
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hooks.append(
            (
                node.in_hook,
                make_input_construction_hook(
                    activation_difference,
                    in_graph_matrix[:prev_index, bwd_index],
                    neuron_matrix,
                ),
            )
        )
    return hooks


def evaluate_graph_with_noise(
    model,
    graph,
    df: pd.DataFrame,
    metrics: Sequence[Callable[..., torch.Tensor]],
    batch_size: int,
    sigma: float,
    hook_name: str,
) -> list[torch.Tensor]:
    from einops import einsum
    from eap.utils import make_hooks_and_matrices, tokenize_plus

    assert model.cfg.use_attn_result, (
        "Model must be configured to use attention result (model.cfg.use_attn_result)"
    )
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, (
            "Model must be configured to ungroup grouped attention "
            "(model.cfg.ungroup_grouped_attention)"
        )

    graph.prune()
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(
            in_graph_matrix,
            node_fully_in_graph,
            "forward backward, forward -> forward backward",
        )
    else:
        neuron_matrix = None

    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix

    results = [[] for _ in metrics]
    batches = list(_iter_batch_frames(df, batch_size))
    for batch_df in tqdm(batches, desc="Noise control budget eval", leave=False):
        clean = batch_df["clean_prefix"].tolist()
        corrupted = batch_df["corrupt_prefix"].tolist()
        label = torch.tensor(
            batch_df["corrupt_metric"].to_numpy(dtype=np.float32),
            device=model.cfg.device,
            dtype=torch.float32,
        )
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(
            model,
            graph,
            len(clean),
            n_pos,
            None,
        )
        input_construction_hooks = _make_input_construction_hooks(
            model=model,
            graph=graph,
            activation_difference=activation_difference,
            in_graph_matrix=in_graph_matrix,
            neuron_matrix=neuron_matrix,
        )
        clean_noise_hooks, corrupt_noise_hooks = make_batch_noise_hooks(
            batch_df=batch_df,
            hook_name=hook_name,
            d_model=model.cfg.d_model,
            sigma=sigma,
            device=model.cfg.device,
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=corrupt_noise_hooks + fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)
            with model.hooks(fwd_hooks=clean_noise_hooks):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)
            with model.hooks(fwd_hooks=clean_noise_hooks + fwd_hooks_clean + input_construction_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)

        for idx, metric in enumerate(metrics):
            values = metric(logits, clean_logits, input_lengths, label).detach().cpu()
            if values.ndim == 0:
                values = values.unsqueeze(0)
            results[idx].append(values)

    return [torch.cat(parts) for parts in results]


def attribute_graph_with_noise(
    model,
    graph,
    df: pd.DataFrame,
    metric: Callable[..., torch.Tensor],
    batch_size: int,
    ig_steps: int,
    sigma: float,
    hook_name: str,
):
    from eap.utils import make_hooks_and_matrices, tokenize_plus

    scores = torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
    )

    total_items = 0
    batches = list(_iter_batch_frames(df, batch_size))
    for batch_df in tqdm(batches, desc="Noisy EAP-IG attribution", leave=False):
        clean = batch_df["clean_prefix"].tolist()
        corrupted = batch_df["corrupt_prefix"].tolist()
        label = torch.tensor(
            batch_df["corrupt_metric"].to_numpy(dtype=np.float32),
            device=model.cfg.device,
            dtype=torch.float32,
        )
        batch_size_value = len(clean)
        total_items += batch_size_value

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, n_pos_corrupted = tokenize_plus(model, corrupted)
        if n_pos != n_pos_corrupted:
            raise ValueError("Clean and corrupt token lengths differ inside the noisy control batch.")

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model,
            graph,
            batch_size_value,
            n_pos,
            scores,
        )
        clean_noise_hooks, corrupt_noise_hooks = make_batch_noise_hooks(
            batch_df=batch_df,
            hook_name=hook_name,
            d_model=model.cfg.d_model,
            sigma=sigma,
            device=model.cfg.device,
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=corrupt_noise_hooks + fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)
            input_acts_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes["input"])].clone()
            with model.hooks(fwd_hooks=clean_noise_hooks + fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)
            input_acts_clean = input_acts_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]

        def input_interpolation_hook(step: int):
            def hook_fn(activations, hook):
                updated = input_acts_corrupted + (step / ig_steps) * (input_acts_clean - input_acts_corrupted)
                updated.requires_grad = True
                return updated

            return hook_fn

        for step in range(ig_steps):
            with model.hooks(
                fwd_hooks=clean_noise_hooks + [(graph.nodes["input"].out_hook, input_interpolation_hook(step))],
                bwd_hooks=bwd_hooks,
            ):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= ig_steps
    graph.scores[:] = scores.to(graph.scores.device)
    return graph


def _baseline_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary = task_accuracy_summary(df)
    summary["absolute_margin_mean"] = float((df["clean_metric"] - df["corrupt_metric"]).abs().mean())
    return summary


def _ambiguity_summary(
    *,
    control_name: str,
    cross_eval: pd.DataFrame,
    original_baseline: dict[str, Any],
    control_baseline: dict[str, Any],
    relative_margin_threshold: float,
    faithfulness_threshold: float,
) -> dict[str, Any]:
    original_margin = float(original_baseline.get("margin_mean", 0.0))
    control_margin = float(control_baseline.get("margin_mean", 0.0))
    margin_ratio = (
        abs(control_margin) / abs(original_margin)
        if abs(original_margin) > 1e-9
        else float("inf")
    )
    max_faithfulness = float(cross_eval["faithfulness_mean"].max()) if not cross_eval.empty else 0.0
    strong_baseline = margin_ratio >= float(relative_margin_threshold)
    strong_cross_eval = max_faithfulness >= float(faithfulness_threshold)
    return {
        "control_type": control_name,
        "baseline_margin_ratio": float(margin_ratio),
        "relative_margin_threshold": float(relative_margin_threshold),
        "max_faithfulness_mean": float(max_faithfulness),
        "faithfulness_threshold": float(faithfulness_threshold),
        "baseline_signal_substantial": bool(strong_baseline),
        "cross_eval_strong": bool(strong_cross_eval),
        "ambiguous": bool(strong_baseline or strong_cross_eval),
    }


def _second_stage_output_dir(control_dir: Path) -> Path:
    return ensure_dir(control_dir / "second_stage_discovery")


def _save_second_stage_artifact(
    *,
    output_dir: Path,
    ranked_edges: Sequence[dict[str, Any]],
    ranked_nodes: Sequence[dict[str, Any]],
    budget_frame: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    edge_path = output_dir / "control_edges.csv"
    node_path = output_dir / "control_nodes.csv"
    budget_path = output_dir / "control_budget_sweep.csv"
    summary_path = output_dir / "control_summary.json"
    save_csv(ranking_frame(ranked_edges), edge_path, index=False)
    save_csv(ranking_frame(ranked_nodes), node_path, index=False)
    save_csv(budget_frame, budget_path, index=False)
    save_json(summary_path, summary)
    return {
        "edge_rankings": str(edge_path),
        "node_rankings": str(node_path),
        "budget_sweep": str(budget_path),
        "summary": str(summary_path),
    }


def run_preposition_control(
    config: PrepositionControlConfig,
    main_artifact_or_rankings: str | Path | None = None,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    prepared = load_control_prepared_inputs(
        project_root=project_root,
        model_name=config.model_name,
        dataset_filter_model_name=config.dataset_filter_model_name,
        target_filter_policy=config.target_filter_policy,
    )
    main_artifact = resolve_main_experiment_artifact(
        project_root=project_root,
        model_name=config.model_name,
        main_artifact_or_rankings=main_artifact_or_rankings or config.main_experiment_path,
    )
    settings = resolve_main_experiment_settings(config, prepared, main_artifact)
    control_dir = control_output_dir(project_root, config.model_name, config.output_day, "by_to_near")
    preview_df = build_preposition_control_dataframe(
        prepared["filtered_df"].head(200),
        prepared["tokenizer"],
        replacement_from=config.replacement_from,
        replacement_to=config.replacement_to,
    )
    preview_path = control_dir / "control_dataset_preview.csv"
    save_csv(preview_df, preview_path, index=False)

    validation_control = build_preposition_control_dataframe(
        settings["validation_df"],
        prepared["tokenizer"],
        replacement_from=config.replacement_from,
        replacement_to=config.replacement_to,
    )
    scored_validation_control = compute_sequence_metrics(
        add_sequence_lengths(validation_control, prepared["model"]),
        model=prepared["model"],
        tokenizer=prepared["tokenizer"],
        animate_ids_tensor=prepared["animate_ids_tensor"],
        inanimate_ids_tensor=prepared["inanimate_ids_tensor"],
        batch_size=config.filter_batch_size,
    )
    scored_validation_control = add_source_pair_keys(scored_validation_control)
    validation_control = require_exact_split_match(
        scored_validation_control,
        settings["validation_df"],
        "Validation split",
    )
    discovery_control = build_preposition_control_dataframe(
        settings["discovery_df"],
        prepared["tokenizer"],
        replacement_from=config.replacement_from,
        replacement_to=config.replacement_to,
    )
    discovery_control = require_exact_split_match(
        discovery_control,
        settings["discovery_df"],
        "Discovery split",
    )
    original_validation = add_source_pair_keys(settings["validation_df"])

    budgets = resolve_main_budgets(
        main_artifact,
        len(main_artifact["ranked_edges"]),
        max_budgets=config.max_budgets,
    )
    validation_loader = make_dataloader(
        validation_control,
        batch_size=settings["evaluation_batch_size"],
        shuffle=False,
    )
    metrics = make_control_eap_metrics(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )
    budget_frame, _ = run_eap_budget_sweep(
        model=prepared["model"],
        scored_graph=build_graph(prepared["model"]),
        ranked_edges=main_artifact["ranked_edges"],
        validation_loader=validation_loader,
        faithfulness_metric=metrics["faithfulness"],
        accuracy_metric=metrics["accuracy"],
        budgets=budgets,
        checkpoint_path=None,
    )
    budget_path = control_dir / "budget_cross_eval.csv"
    save_csv(budget_frame, budget_path, index=False)

    original_baseline = _baseline_summary(original_validation)
    control_baseline = _baseline_summary(validation_control)
    baseline_comparison = {
        "control_type": "by_to_near",
        "replacement_from": config.replacement_from,
        "replacement_to": config.replacement_to,
        "target_filtered_count": int(len(prepared["filtered_df"])),
        "validation_count": int(len(validation_control)),
        "original_validation_baseline": original_baseline,
        "control_validation_baseline": control_baseline,
    }
    baseline_path = control_dir / "baseline_comparison.json"
    save_json(baseline_path, baseline_comparison)

    ambiguity = _ambiguity_summary(
        control_name="by_to_near",
        cross_eval=budget_frame,
        original_baseline=original_baseline,
        control_baseline=control_baseline,
        relative_margin_threshold=config.ambiguity_relative_margin_threshold,
        faithfulness_threshold=config.ambiguity_faithfulness_threshold,
    )
    second_stage_paths = None
    if ambiguity["ambiguous"] and config.run_second_stage_discovery_on_ambiguous:
        discovery_loader = make_dataloader(
            discovery_control,
            batch_size=settings["attribution_batch_size"],
            shuffle=False,
        )
        scored_graph = attribute_graph(
            model=prepared["model"],
            graph=build_graph(prepared["model"]),
            dataloader=discovery_loader,
            metric=metrics["attribute"],
            ig_steps=settings["ig_steps"],
        )
        ranked_edges = collapsed_edge_groups(scored_graph)
        ranked_nodes = induced_node_ranking(ranked_edges)
        resolved_budgets = resolve_main_budgets(
            main_artifact,
            len(ranked_edges),
            max_budgets=config.max_budgets,
        )
        control_budget_frame, _ = run_eap_budget_sweep(
            model=prepared["model"],
            scored_graph=scored_graph,
            ranked_edges=ranked_edges,
            validation_loader=validation_loader,
            faithfulness_metric=metrics["faithfulness"],
            accuracy_metric=metrics["accuracy"],
            budgets=resolved_budgets,
            checkpoint_path=None,
        )
        second_stage_summary = {
            "control_type": "by_to_near",
            "discovery_count": int(len(discovery_control)),
            "validation_count": int(len(validation_control)),
            "resolved_budget_grid": resolved_budgets,
        }
        second_stage_paths = _save_second_stage_artifact(
            output_dir=_second_stage_output_dir(control_dir),
            ranked_edges=ranked_edges,
            ranked_nodes=ranked_nodes,
            budget_frame=control_budget_frame,
            summary=second_stage_summary,
        )

    control_summary = {
        "config": asdict(config),
        "main_artifact": {
            "summary_path": main_artifact["summary_path"],
            "edge_path": main_artifact["edge_path"],
            "budget_path": main_artifact["budget_path"],
        },
        "paths": {
            "output_dir": str(control_dir),
            "control_dataset_preview": str(preview_path),
            "baseline_comparison": str(baseline_path),
            "budget_cross_eval": str(budget_path),
            "second_stage_discovery": second_stage_paths,
        },
        "dataset_summary": {
            "target_filtered_count": int(len(prepared["filtered_df"])),
            "discovery_count": int(len(settings["discovery_df"])),
            "validation_count": int(len(settings["validation_df"])),
        },
        "ambiguity": ambiguity,
    }
    summary_path = control_dir / "control_summary.json"
    save_json(summary_path, control_summary)
    control_summary["paths"]["control_summary"] = str(summary_path)
    save_json(summary_path, control_summary)
    return control_summary


def run_verb_noise_control(
    config: VerbNoiseControlConfig,
    main_artifact_or_rankings: str | Path | None = None,
    sigma: float | None = None,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    prepared = load_control_prepared_inputs(
        project_root=project_root,
        model_name=config.model_name,
        dataset_filter_model_name=config.dataset_filter_model_name,
        target_filter_policy=config.target_filter_policy,
    )
    main_artifact = resolve_main_experiment_artifact(
        project_root=project_root,
        model_name=config.model_name,
        main_artifact_or_rankings=main_artifact_or_rankings or config.main_experiment_path,
    )
    settings = resolve_main_experiment_settings(config, prepared, main_artifact)
    control_dir = control_output_dir(project_root, config.model_name, config.output_day, "verb_noise")
    sigma_value = float(config.sigma if sigma is None else sigma)
    hook_name = concept_hook_name(0, config.noise_site)

    validation_control = prepare_verb_noise_control_dataframe(
        settings["validation_df"],
        model=prepared["model"],
        tokenizer=prepared["tokenizer"],
        seed=config.seed,
    )
    scored_validation_control = compute_noisy_sequence_metrics(
        df=validation_control,
        model=prepared["model"],
        animate_ids_tensor=prepared["animate_ids_tensor"],
        inanimate_ids_tensor=prepared["inanimate_ids_tensor"],
        batch_size=config.filter_batch_size,
        sigma=sigma_value,
        hook_name=hook_name,
    )
    validation_control = require_exact_split_match(
        scored_validation_control,
        settings["validation_df"],
        "Validation split",
    )
    original_validation = add_source_pair_keys(settings["validation_df"])
    budgets = resolve_main_budgets(
        main_artifact,
        len(main_artifact["ranked_edges"]),
        max_budgets=config.max_budgets,
    )

    metrics = make_control_eap_metrics(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )
    scored_graph = build_graph(prepared["model"])
    rows: list[dict[str, Any]] = []
    for budget in tqdm(budgets, desc="Noise control budget sweep"):
        candidate_graph = build_budget_circuit(scored_graph, main_artifact["ranked_edges"], int(budget))
        faithfulness, accuracy = evaluate_graph_with_noise(
            model=prepared["model"],
            graph=candidate_graph,
            df=validation_control,
            metrics=[metrics["faithfulness"], metrics["accuracy"]],
            batch_size=settings["evaluation_batch_size"],
            sigma=sigma_value,
            hook_name=hook_name,
        )
        rows.append(
            {
                "collapsed_edge_budget": int(budget),
                "budget_fraction": float(budget / len(main_artifact["ranked_edges"])) if main_artifact["ranked_edges"] else 0.0,
                "expanded_edge_count": int(candidate_graph.count_included_edges()),
                "induced_node_count": int(candidate_graph.count_included_nodes() - 2),
                "faithfulness_mean": float(faithfulness.mean().item()),
                "faithfulness_std": float(faithfulness.std(unbiased=False).item()) if len(faithfulness) > 1 else 0.0,
                "accuracy_mean": float(accuracy.mean().item()),
                "accuracy_std": float(accuracy.std(unbiased=False).item()) if len(accuracy) > 1 else 0.0,
                "validation_examples": int(len(faithfulness)),
            }
        )
    budget_frame = pd.DataFrame(rows)
    budget_path = control_dir / "budget_cross_eval.csv"
    save_csv(budget_frame, budget_path, index=False)

    original_baseline = _baseline_summary(original_validation)
    control_baseline = _baseline_summary(validation_control)
    baseline_comparison = {
        "control_type": "verb_noise",
        "noise_site": hook_name,
        "sigma": float(sigma_value),
        "target_filtered_count": int(len(prepared["filtered_df"])),
        "validation_count": int(len(validation_control)),
        "original_validation_baseline": original_baseline,
        "control_validation_baseline": control_baseline,
    }
    baseline_path = control_dir / "baseline_comparison.json"
    save_json(baseline_path, baseline_comparison)

    ambiguity = _ambiguity_summary(
        control_name="verb_noise",
        cross_eval=budget_frame,
        original_baseline=original_baseline,
        control_baseline=control_baseline,
        relative_margin_threshold=config.ambiguity_relative_margin_threshold,
        faithfulness_threshold=config.ambiguity_faithfulness_threshold,
    )
    second_stage_paths = None
    if ambiguity["ambiguous"] and config.run_second_stage_discovery_on_ambiguous:
        discovery_control = prepare_verb_noise_control_dataframe(
            settings["discovery_df"],
            model=prepared["model"],
            tokenizer=prepared["tokenizer"],
            seed=config.seed,
        )
        discovery_control = require_exact_split_match(
            discovery_control,
            settings["discovery_df"],
            "Discovery split",
        )
        scored_graph = attribute_graph_with_noise(
            model=prepared["model"],
            graph=build_graph(prepared["model"]),
            df=discovery_control,
            metric=metrics["attribute"],
            batch_size=settings["attribution_batch_size"],
            ig_steps=settings["ig_steps"],
            sigma=sigma_value,
            hook_name=hook_name,
        )
        ranked_edges = collapsed_edge_groups(scored_graph)
        ranked_nodes = induced_node_ranking(ranked_edges)
        resolved_budgets = resolve_main_budgets(
            main_artifact,
            len(ranked_edges),
            max_budgets=config.max_budgets,
        )
        stage_rows: list[dict[str, Any]] = []
        for budget in tqdm(resolved_budgets, desc="Noisy second-stage budget sweep"):
            candidate_graph = build_budget_circuit(scored_graph, ranked_edges, int(budget))
            faithfulness, accuracy = evaluate_graph_with_noise(
                model=prepared["model"],
                graph=candidate_graph,
                df=validation_control,
                metrics=[metrics["faithfulness"], metrics["accuracy"]],
                batch_size=settings["evaluation_batch_size"],
                sigma=sigma_value,
                hook_name=hook_name,
            )
            stage_rows.append(
                {
                    "collapsed_edge_budget": int(budget),
                    "budget_fraction": float(budget / len(ranked_edges)) if ranked_edges else 0.0,
                    "expanded_edge_count": int(candidate_graph.count_included_edges()),
                    "induced_node_count": int(candidate_graph.count_included_nodes() - 2),
                    "faithfulness_mean": float(faithfulness.mean().item()),
                    "faithfulness_std": float(faithfulness.std(unbiased=False).item()) if len(faithfulness) > 1 else 0.0,
                    "accuracy_mean": float(accuracy.mean().item()),
                    "accuracy_std": float(accuracy.std(unbiased=False).item()) if len(accuracy) > 1 else 0.0,
                    "validation_examples": int(len(faithfulness)),
                }
            )
        second_stage_paths = _save_second_stage_artifact(
            output_dir=_second_stage_output_dir(control_dir),
            ranked_edges=ranked_edges,
            ranked_nodes=ranked_nodes,
            budget_frame=pd.DataFrame(stage_rows),
            summary={
                "control_type": "verb_noise",
                "sigma": float(sigma_value),
                "noise_site": hook_name,
                "discovery_count": int(len(discovery_control)),
                "validation_count": int(len(validation_control)),
                "resolved_budget_grid": resolved_budgets,
            },
        )

    control_summary = {
        "config": asdict(config),
        "main_artifact": {
            "summary_path": main_artifact["summary_path"],
            "edge_path": main_artifact["edge_path"],
            "budget_path": main_artifact["budget_path"],
        },
        "paths": {
            "output_dir": str(control_dir),
            "baseline_comparison": str(baseline_path),
            "budget_cross_eval": str(budget_path),
            "second_stage_discovery": second_stage_paths,
        },
        "dataset_summary": {
            "target_filtered_count": int(len(prepared["filtered_df"])),
            "discovery_count": int(len(settings["discovery_df"])),
            "validation_count": int(len(settings["validation_df"])),
        },
        "sigma": float(sigma_value),
        "noise_site": hook_name,
        "ambiguity": ambiguity,
    }
    summary_path = control_dir / "control_summary.json"
    save_json(summary_path, control_summary)
    control_summary["paths"]["control_summary"] = str(summary_path)
    save_json(summary_path, control_summary)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return control_summary


def run_blimp_passive_prefix_control(
    config: BlimpPassivePrefixControlConfig,
    main_artifact_or_rankings: str | Path | None = None,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    resolved_model_name = canonical_model_name(config.model_name)
    context = load_model_context(project_root, resolved_model_name)
    model = context["model"]
    tokenizer = context["tokenizer"]
    main_artifact = resolve_main_experiment_artifact(
        project_root=project_root,
        model_name=resolved_model_name,
        main_artifact_or_rankings=main_artifact_or_rankings or config.main_experiment_path,
    )

    if config.budget is None:
        if main_artifact["budget_frame"].empty:
            raise ValueError("Source full-model budget sweep is required to resolve the 85% circuit budget.")
        threshold_row = first_budget_reaching_faithfulness(
            main_artifact["budget_frame"],
            float(config.source_faithfulness_threshold),
        )
        selected_budget = int(threshold_row["collapsed_edge_budget"])
        selected_budget_row = dict(threshold_row)
    else:
        threshold_row = None
        selected_budget = int(config.budget)
        selected_budget_row = None

    candidate_graph = build_budget_circuit(
        build_graph(model),
        main_artifact["ranked_edges"],
        selected_budget,
    )

    raw_df, dataset_path = load_local_blimp_prefix_dataset(
        project_root,
        "animate_subject_passive",
    )
    prepared_rows, prefix_failures = prepare_blimp_passive_prefix_rows(raw_df, tokenizer)
    result_rows = evaluate_blimp_passive_prefix_control(
        model=model,
        graph=candidate_graph,
        df=prepared_rows,
        animate_ids_tensor=context["animate_ids_tensor"],
        inanimate_ids_tensor=context["inanimate_ids_tensor"],
        batch_size=int(config.evaluation_batch_size),
    )
    summary = summarize_blimp_passive_prefix_control(result_rows)

    control_dir = control_output_dir(project_root, resolved_model_name, config.output_day, "blimp_passive_prefix")
    rows_path = control_dir / "rows.csv"
    summary_path = control_dir / "summary.json"
    status_path = control_dir / "status.json"
    failures_path = control_dir / "prefix_failures.csv"
    save_csv(result_rows, rows_path, index=False)
    save_csv(prefix_failures, failures_path, index=False)

    status = {
        "config": asdict(config),
        "main_artifact": {
            "summary_path": main_artifact["summary_path"],
            "edge_path": main_artifact["edge_path"],
            "budget_path": main_artifact["budget_path"],
        },
        "dataset_source": str(dataset_path),
        "selected_budget": int(selected_budget),
        "selected_budget_row": selected_budget_row,
        "dataset_summary": {
            "raw_rows": int(len(raw_df)),
            "valid_rows": int(len(prepared_rows)),
            "filtered_prefix_rows": int(len(prefix_failures)),
            "expanded_edge_count": int(candidate_graph.count_included_edges()),
            "induced_node_count": int(candidate_graph.count_included_nodes() - 2),
            "animate_target_count": int(len(context["animate_ids_tensor"])),
            "inanimate_target_count": int(len(context["inanimate_ids_tensor"])),
        },
        "target_sets": context["target_tokenization_diagnostics"],
        "paths": {
            "output_dir": str(control_dir),
            "rows": str(rows_path),
            "summary": str(summary_path),
            "status": str(status_path),
            "prefix_failures": str(failures_path),
        },
    }
    save_json(summary_path, summary)
    save_json(status_path, status)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": status,
        "summary": summary,
        "paths": status["paths"],
    }

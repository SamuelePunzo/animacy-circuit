from __future__ import annotations

import argparse
import gc
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from circuit_finder_core import (
    add_sequence_lengths,
    component_token_span,
    eap_node_metadata,
    filter_df_to_prompt_pairs,
    generate_exact_length_batches,
    intersect_prompt_pair_frames,
    load_model_context,
    load_policy_filtered_dataset,
    prompt_pair_columns,
    resolve_animacy_circuit_root,
)

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


SLOT_COLUMNS = ["model_slug", "run_name", "sample_size", "seed"]
STRUCTURAL_NODES = {"input", "logits"}
MODEL_SLUG_OVERRIDES = {
    "gpt2": "gpt2",
    "Qwen_Qwen3-4B": "Qwen/Qwen3-4B",
    "google_gemma-3-4b-pt": "google/gemma-3-4b-pt",
    "meta-llama_Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
}
DEFAULT_BATCH_SIZE_BY_MODEL_SLUG = {
    "gpt2": 128,
    "meta-llama_Llama-3.2-3B": 128,
    "google_gemma-3-4b-pt": 72,
    "Qwen_Qwen3-4B": 32,
}
MODEL_PROCESS_ORDER = {
    "gpt2": 0,
    "google_gemma-3-4b-pt": 1,
    "meta-llama_Llama-3.2-3B": 2,
    "Qwen_Qwen3-4B": 3,
}


def latest_report_dir(results_root: Path) -> Path:
    preferred = sorted(results_root.glob("necessary_edge_expansion_main_original_20_50_*"))
    fallback = sorted(results_root.glob("necessary_edge_expansion_*"))
    candidates = preferred or fallback
    if not candidates:
        raise FileNotFoundError(f"No necessary-edge reports found under {results_root}")
    return candidates[-1]


def model_name_from_slug(model_slug: str) -> str:
    if model_slug in MODEL_SLUG_OVERRIDES:
        return MODEL_SLUG_OVERRIDES[model_slug]
    if "_" in model_slug:
        return model_slug.replace("_", "/", 1)
    return model_slug


def batch_size_for_model(model_slug: str, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    return int(DEFAULT_BATCH_SIZE_BY_MODEL_SLUG.get(model_slug, 4))


def sorted_slots(selected_summary: pd.DataFrame) -> pd.DataFrame:
    rows = selected_summary.copy()
    rows["_process_order"] = rows["model_slug"].map(MODEL_PROCESS_ORDER).fillna(100).astype(int)
    return rows.sort_values(["_process_order", *SLOT_COLUMNS], kind="stable").drop(columns=["_process_order"])


def resolve_artifact_path(project_root: Path, value: str | Path | float | None) -> Path | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    for root in (project_root.parent, project_root, Path.cwd().resolve()):
        candidate = root / path
        if candidate.exists():
            return candidate
    return project_root.parent / path


def localization_summary_path(project_root: Path, slot_row: pd.Series | dict[str, Any]) -> Path | None:
    edge_path = resolve_artifact_path(project_root, slot_row.get("edge_path")) if hasattr(slot_row, "get") else None
    if edge_path is None or not edge_path.exists():
        return None
    sample_size = int(slot_row["sample_size"])
    seed = int(slot_row["seed"])
    candidate = edge_path.with_name(f"localization_summary_sample_{sample_size}_seed_{seed}.json")
    return candidate if candidate.exists() else None


def slot_run_metadata(project_root: Path, slot_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    model_slug = str(slot_row["model_slug"])
    metadata = {
        "model_slug": model_slug,
        "model_name": model_name_from_slug(model_slug),
        "target_source": None,
        "target_filter_policy": "model_success",
        "target_settings_source": "repo_default",
        "summary_path": None,
    }
    path = localization_summary_path(project_root, slot_row)
    if path is None:
        return metadata
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {}) or {}
    dataset_summary = payload.get("dataset_summary", {}) or {}
    metadata.update(
        {
            "model_name": dataset_summary.get("target_model") or config.get("model_name") or metadata["model_name"],
            "target_source": config.get("target_source") or payload.get("target_source"),
            "target_filter_policy": config.get("target_filter_policy", metadata["target_filter_policy"]),
            "summary_path": str(path),
        }
    )
    if metadata["target_source"] is not None:
        metadata["target_settings_source"] = "localization_summary"
    return metadata


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def role_for_edge(parent_type: str, child_type: str) -> str:
    if parent_type == "input" and child_type in {"attn", "mlp"}:
        return "input_to_component"
    if parent_type == "attn" and child_type == "mlp":
        return "head_to_mlp"
    if parent_type == "mlp" and child_type == "mlp":
        return "mlp_to_mlp"
    if child_type == "logits" and parent_type in {"attn", "mlp"}:
        return "component_to_logits"
    return "other"


def add_edge_semantics(edges: pd.DataFrame) -> pd.DataFrame:
    rows = edges.copy().reset_index(drop=True)
    parent_meta = [eap_node_metadata(str(node)) for node in rows["parent"]]
    child_meta = [eap_node_metadata(str(node)) for node in rows["child"]]
    rows["parent_type"] = [meta["kind"] for meta in parent_meta]
    rows["child_type"] = [meta["kind"] for meta in child_meta]
    rows["parent_layer"] = [meta["layer"] for meta in parent_meta]
    rows["child_layer"] = [meta["layer"] for meta in child_meta]
    rows["edge_role"] = [
        role_for_edge(parent_type, child_type)
        for parent_type, child_type in zip(rows["parent_type"], rows["child_type"])
    ]
    return rows


def component_cards_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for edge in edges.to_dict("records"):
        for endpoint, direction in ((edge["parent"], "outgoing"), (edge["child"], "incoming")):
            meta = eap_node_metadata(str(endpoint))
            if meta["kind"] in STRUCTURAL_NODES:
                continue
            rows.append(
                {
                    **{col: edge[col] for col in SLOT_COLUMNS + ["selected_budget"]},
                    "component": str(endpoint),
                    "component_type": meta["kind"],
                    "layer": int(meta["layer"]),
                    "head": meta["head"],
                    "direction": direction,
                    "edge_role": edge["edge_role"],
                    "collapsed_edge": edge["collapsed_edge"],
                    "rank": edge["rank"],
                    "signed_sum": edge["signed_sum"],
                    "abs_score": edge["abs_score"],
                }
            )
    incidents = pd.DataFrame(rows)
    if incidents.empty:
        return pd.DataFrame()
    return (
        incidents.groupby(SLOT_COLUMNS + ["selected_budget", "component", "component_type", "layer", "head"], dropna=False)
        .agg(
            incident_edge_count=("collapsed_edge", "nunique"),
            abs_incident_mass=("abs_score", "sum"),
            signed_incident_mass=("signed_sum", "sum"),
            best_rank=("rank", "min"),
        )
        .reset_index()
    )


def component_hook_spec(component: str) -> dict[str, Any]:
    meta = eap_node_metadata(component)
    if meta["kind"] == "mlp":
        return {
            "component": component,
            "component_type": "mlp",
            "layer": int(meta["layer"]),
            "head": None,
            "activation_hook": f"blocks.{int(meta['layer'])}.hook_mlp_out",
            "pattern_hook": None,
        }
    if meta["kind"] == "attn":
        return {
            "component": component,
            "component_type": "attn",
            "layer": int(meta["layer"]),
            "head": int(meta["head"]),
            "activation_hook": f"blocks.{int(meta['layer'])}.attn.hook_result",
            "pattern_hook": f"blocks.{int(meta['layer'])}.attn.hook_pattern",
        }
    return {
        "component": component,
        "component_type": meta["kind"],
        "layer": meta["layer"],
        "head": meta["head"],
        "activation_hook": None,
        "pattern_hook": None,
    }


def hooks_for_components(components: list[str], include_patterns: bool = True) -> list[str]:
    hooks = []
    for component in components:
        spec = component_hook_spec(component)
        if spec["activation_hook"]:
            hooks.append(spec["activation_hook"])
        if include_patterns and spec["pattern_hook"]:
            hooks.append(spec["pattern_hook"])
    return sorted(set(hooks))


def normalize_prompt_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.copy()
    if "clean_prefix" not in rows.columns and "clean" in rows.columns:
        rows["clean_prefix"] = rows["clean"]
    if "corrupt_prefix" not in rows.columns and "corrupt" in rows.columns:
        rows["corrupt_prefix"] = rows["corrupt"]
    missing = sorted({"clean_prefix", "corrupt_prefix"} - set(rows.columns))
    if missing:
        raise ValueError(f"Dataset is missing prompt columns: {missing}")
    return rows


def sample_examples(df: pd.DataFrame, n_examples: int, seed: int) -> pd.DataFrame:
    if n_examples is None or n_examples >= len(df):
        return df.reset_index(drop=True).copy()
    return df.sample(n=n_examples, random_state=seed).reset_index(drop=True).copy()


def load_slot_examples(
    project_root: Path,
    slot_row: pd.Series,
    n_examples: int,
    batch_size: int,
    prompt_pairs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = slot_run_metadata(project_root, slot_row)
    context = load_model_context(project_root, metadata["model_name"], target_source=metadata["target_source"])
    model = context["model"]
    raw_df = load_policy_filtered_dataset(
        project_root=project_root,
        model_name=metadata["model_name"],
        batch_size=batch_size,
        target_filter_policy=metadata["target_filter_policy"],
        cache=True,
        max_examples=None,
        seed=int(slot_row["seed"]),
        target_source=metadata["target_source"],
    )
    raw_df = normalize_prompt_columns(raw_df)
    if prompt_pairs is not None and not prompt_pairs.empty:
        raw_df = filter_df_to_prompt_pairs(raw_df, prompt_pairs)
    sampled = sample_examples(raw_df, n_examples=n_examples, seed=int(slot_row["seed"]))
    sampled = add_sequence_lengths(sampled, model)
    if sampled.empty:
        raise ValueError("No same-length clean/corrupt examples survived tokenization filtering for this slot.")
    return sampled, {**context, **metadata}


def single_token_position(tokenizer, text: str, token_text: str | None) -> int | None:
    if token_text is None or pd.isna(token_text):
        return None
    span, error = component_token_span(tokenizer, str(text), str(token_text))
    if span is None or error is not None or span[1] - span[0] != 1:
        return None
    return int(span[0] + 1)


def add_key_positions(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows = df.reset_index(drop=True).copy()
    clean_initial_the_positions = []
    corrupt_initial_the_positions = []
    clean_patient_positions = []
    corrupt_patient_positions = []
    clean_was_positions = []
    corrupt_was_positions = []
    clean_verb_positions = []
    corrupt_verb_positions = []
    clean_by_positions = []
    corrupt_by_positions = []
    for row in rows.to_dict("records"):
        patient = row.get("patient", row.get("patient_x", row.get("patient_y")))
        clean_verb = row.get("clean_verb", row.get("clean_verb_x", row.get("clean_verb_y")))
        corrupt_verb = row.get("corrupt_verb", row.get("corrupt_verb_x", row.get("corrupt_verb_y")))
        clean_initial_the_positions.append(single_token_position(tokenizer, row["clean_prefix"], "The"))
        corrupt_initial_the_positions.append(single_token_position(tokenizer, row["corrupt_prefix"], "The"))
        clean_patient_positions.append(single_token_position(tokenizer, row["clean_prefix"], patient))
        corrupt_patient_positions.append(single_token_position(tokenizer, row["corrupt_prefix"], patient))
        clean_was_positions.append(single_token_position(tokenizer, row["clean_prefix"], "was"))
        corrupt_was_positions.append(single_token_position(tokenizer, row["corrupt_prefix"], "was"))
        clean_verb_positions.append(single_token_position(tokenizer, row["clean_prefix"], clean_verb))
        corrupt_verb_positions.append(single_token_position(tokenizer, row["corrupt_prefix"], corrupt_verb))
        clean_by_positions.append(single_token_position(tokenizer, row["clean_prefix"], "by"))
        corrupt_by_positions.append(single_token_position(tokenizer, row["corrupt_prefix"], "by"))
    rows["bos_pos"] = 0
    rows["final_pos"] = rows["seq_len"].astype(int) - 1
    rows["clean_initial_the_pos"] = clean_initial_the_positions
    rows["corrupt_initial_the_pos"] = corrupt_initial_the_positions
    rows["clean_patient_pos"] = clean_patient_positions
    rows["corrupt_patient_pos"] = corrupt_patient_positions
    rows["clean_was_pos"] = clean_was_positions
    rows["corrupt_was_pos"] = corrupt_was_positions
    rows["clean_verb_pos"] = clean_verb_positions
    rows["corrupt_verb_pos"] = corrupt_verb_positions
    rows["clean_by_pos"] = clean_by_positions
    rows["corrupt_by_pos"] = corrupt_by_positions
    return rows


def unembedding_matrix(model):
    if hasattr(model, "W_U"):
        return model.W_U
    if hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        return model.unembed.W_U
    raise AttributeError("Could not locate model unembedding matrix W_U.")


def animacy_readout_direction(model, animate_ids_tensor, inanimate_ids_tensor):
    w_u = unembedding_matrix(model)
    animate_ids = animate_ids_tensor.to(device=w_u.device, dtype=torch.long)
    inanimate_ids = inanimate_ids_tensor.to(device=w_u.device, dtype=torch.long)
    return w_u[:, animate_ids].mean(dim=1) - w_u[:, inanimate_ids].mean(dim=1)


def nearest_centroid_accuracy(features: np.ndarray, labels: np.ndarray) -> float | None:
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        return None
    centroids = {label: features[labels == label].mean(axis=0) for label in unique}
    correct = 0
    for vector, label in zip(features, labels):
        predicted = min(unique, key=lambda candidate: float(np.linalg.norm(vector - centroids[candidate])))
        correct += int(predicted == label)
    return float(correct / len(labels)) if len(labels) else None


def pca_geometry_summary(activation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if PCA is None:
        return pd.DataFrame(), pd.DataFrame([{"status": "skipped", "reason": "sklearn unavailable"}])
    summaries = []
    projections = []
    group_cols = SLOT_COLUMNS + ["component", "component_type", "layer", "head", "position_label"]
    for key, group in activation_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        clean_rows = group[group["condition"] == "clean"]
        corrupt_rows = group[group["condition"] == "corrupt"]
        if len(clean_rows) < 2 or len(corrupt_rows) < 2:
            summaries.append({**key_dict, "status": "skipped", "reason": "fewer than two examples per condition"})
            continue
        clean = np.stack(clean_rows["activation"].to_numpy())
        corrupt = np.stack(corrupt_rows["activation"].to_numpy())
        clean_map = {row["example_id"]: row["activation"] for row in clean_rows.to_dict("records")}
        corrupt_map = {row["example_id"]: row["activation"] for row in corrupt_rows.to_dict("records")}
        paired_ids = sorted(set(clean_map).intersection(corrupt_map))
        delta_norms = np.array(
            [float(np.linalg.norm(clean_map[item] - corrupt_map[item])) for item in paired_ids],
            dtype=float,
        )
        features = np.concatenate([clean, corrupt], axis=0)
        labels = np.array(["clean"] * len(clean) + ["corrupt"] * len(corrupt))
        if features.shape[0] < 3 or features.shape[1] < 2:
            summaries.append({**key_dict, "status": "skipped", "reason": "insufficient PCA dimensions"})
            continue
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(features)
        clean_centroid = xy[labels == "clean"].mean(axis=0)
        corrupt_centroid = xy[labels == "corrupt"].mean(axis=0)
        summaries.append(
            {
                **key_dict,
                "status": "ok",
                "example_count": int(features.shape[0]),
                "pca_explained_variance_1": float(pca.explained_variance_ratio_[0]),
                "pca_explained_variance_2": float(pca.explained_variance_ratio_[1]),
                "clean_corrupt_centroid_distance": float(np.linalg.norm(clean_centroid - corrupt_centroid)),
                "nearest_centroid_accuracy": nearest_centroid_accuracy(xy, labels),
                "paired_delta_count": int(len(delta_norms)),
                "delta_vector_norm_mean": float(delta_norms.mean()) if len(delta_norms) else None,
                "delta_vector_norm_std": float(delta_norms.std()) if len(delta_norms) else None,
            }
        )
        for idx, (x, y) in enumerate(xy):
            projections.append({**key_dict, "condition": labels[idx], "pc1": float(x), "pc2": float(y)})
    return pd.DataFrame(summaries), pd.DataFrame(projections)


def cache_component_activation(cache, spec: dict[str, Any], batch_index: int, position: int):
    value = cache[spec["activation_hook"]]
    if spec["component_type"] == "attn":
        if value.ndim != 4:
            raise ValueError(f"Expected attention result cache with 4 dims for {spec['component']}, got {tuple(value.shape)}")
        return value[batch_index, position, int(spec["head"]), :]
    if value.ndim != 3:
        raise ValueError(f"Expected MLP cache with 3 dims for {spec['component']}, got {tuple(value.shape)}")
    return value[batch_index, position, :]


def append_readout_and_activation_rows(
    rows: list[dict[str, Any]],
    activations: list[dict[str, Any]],
    slot_row: pd.Series,
    component: str,
    spec: dict[str, Any],
    condition: str,
    position_label: str,
    batch_index: int,
    example_id: Any,
    position: int | None,
    cache,
    readout_dir,
) -> None:
    if position is None or pd.isna(position):
        return
    position = int(position)
    vector = cache_component_activation(cache, spec, batch_index, position).detach().float().cpu()
    readout = float(vector.to(readout_dir.device).dot(readout_dir).detach().float().cpu().item())
    base = {col: slot_row[col] for col in SLOT_COLUMNS}
    base.update(
        {
            "selected_budget": int(slot_row["selected_budget"]),
            "component": component,
            "component_type": spec["component_type"],
            "layer": spec["layer"],
            "head": spec["head"],
            "condition": condition,
            "example_id": example_id,
            "position_label": position_label,
            "position": position,
        }
    )
    rows.append({**base, "readout": readout})
    activations.append({**base, "activation": vector.numpy()})


def append_attention_rows(
    rows: list[dict[str, Any]],
    slot_row: pd.Series,
    component: str,
    spec: dict[str, Any],
    condition: str,
    batch_index: int,
    query_position: int,
    key_positions: dict[str, int | None],
    cache,
) -> None:
    pattern_hook = spec.get("pattern_hook")
    if not pattern_hook or pattern_hook not in cache:
        return
    pattern = cache[pattern_hook]
    if pattern.ndim != 4:
        return
    head_pattern = pattern[batch_index, int(spec["head"]), int(query_position), :].detach().float().cpu()
    base = {col: slot_row[col] for col in SLOT_COLUMNS}
    base.update(
        {
            "selected_budget": int(slot_row["selected_budget"]),
            "component": component,
            "component_type": spec["component_type"],
            "layer": spec["layer"],
            "head": spec["head"],
            "condition": condition,
            "query_position_label": "final_token",
            "query_position": int(query_position),
        }
    )
    for label, position in key_positions.items():
        if position is None or pd.isna(position):
            continue
        position = int(position)
        if 0 <= position < head_pattern.shape[0]:
            rows.append(
                {
                    **base,
                    "key_position_label": label,
                    "key_position": position,
                    "attention_mass": float(head_pattern[position].item()),
                }
            )


def classify_readout_row(row: pd.Series, eps: float = 1e-6) -> str:
    values = [row.get("clean"), row.get("corrupt")]
    values = [float(value) for value in values if pd.notna(value)]
    if not values or max(abs(value) for value in values) <= eps:
        return "low_readout"
    signs = {1 if value > eps else -1 if value < -eps else 0 for value in values}
    if signs == {1}:
        return "animate_pushing"
    if signs == {-1}:
        return "inanimate_pushing"
    return "mixed"


def analyze_slot_activations(
    project_root: Path,
    semantic_edges: pd.DataFrame,
    component_cards: pd.DataFrame,
    slot_row: pd.Series,
    n_examples: int,
    batch_size: int,
    prompt_pairs: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    del semantic_edges
    slot_components = component_cards.merge(pd.DataFrame([slot_row[SLOT_COLUMNS].to_dict()]), on=SLOT_COLUMNS, how="inner")
    components = slot_components["component"].dropna().astype(str).drop_duplicates().tolist()
    specs = {component: component_hook_spec(component) for component in components}
    supported_components = [component for component, spec in specs.items() if spec["activation_hook"] is not None]
    if not supported_components:
        return {}

    examples, context = load_slot_examples(project_root, slot_row, n_examples, batch_size, prompt_pairs=prompt_pairs)
    model = context["model"]
    tokenizer = context["tokenizer"]
    examples = add_key_positions(examples, tokenizer)
    readout_dir = animacy_readout_direction(model, context["animate_ids_tensor"], context["inanimate_ids_tensor"])
    hook_names = hooks_for_components(supported_components, include_patterns=True)
    available_hooks = set(name for name, _hook in model.hook_dict.items()) if hasattr(model, "hook_dict") else set(hook_names)
    hook_names = [hook for hook in hook_names if hook in available_hooks]
    unsupported = sorted(set(hooks_for_components(supported_components, include_patterns=True)) - set(hook_names))
    if unsupported:
        print(f"Skipping unsupported hooks for {slot_row['model_slug']} seed {slot_row['seed']}: {unsupported}")

    readout_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    device = model.cfg.device

    for clean_tokens, corrupt_tokens, batch_df in generate_exact_length_batches(examples, model, batch_size=batch_size, device=device):
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_names)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=hook_names)
        for batch_index, (example_id, row) in enumerate(batch_df.iterrows()):
            position_map = {
                "final_token": (row["final_pos"], row["final_pos"]),
                "initial_the_token": (row.get("clean_initial_the_pos"), row.get("corrupt_initial_the_pos")),
                "verb_token": (row.get("clean_verb_pos"), row.get("corrupt_verb_pos")),
                "patient_token": (row.get("clean_patient_pos"), row.get("corrupt_patient_pos")),
                "was_token": (row.get("clean_was_pos"), row.get("corrupt_was_pos")),
                "by_token": (row.get("clean_by_pos"), row.get("corrupt_by_pos")),
            }
            for component in supported_components:
                spec = specs[component]
                if spec["activation_hook"] not in clean_cache or spec["activation_hook"] not in corrupt_cache:
                    continue
                for position_label, (clean_pos, corrupt_pos) in position_map.items():
                    append_readout_and_activation_rows(
                        readout_rows,
                        activation_rows,
                        slot_row,
                        component,
                        spec,
                        "clean",
                        position_label,
                        batch_index,
                        example_id,
                        clean_pos,
                        clean_cache,
                        readout_dir,
                    )
                    append_readout_and_activation_rows(
                        readout_rows,
                        activation_rows,
                        slot_row,
                        component,
                        spec,
                        "corrupt",
                        position_label,
                        batch_index,
                        example_id,
                        corrupt_pos,
                        corrupt_cache,
                        readout_dir,
                    )
                if spec["component_type"] == "attn" and spec.get("pattern_hook") in clean_cache:
                    append_attention_rows(
                        attention_rows,
                        slot_row,
                        component,
                        spec,
                        "clean",
                        batch_index,
                        int(row["final_pos"]),
                        {
                            "BOS": row["bos_pos"],
                            "The": row.get("clean_initial_the_pos"),
                            "patient": row.get("clean_patient_pos"),
                            "was": row.get("clean_was_pos"),
                            "verb": row.get("clean_verb_pos"),
                            "by": row.get("clean_by_pos"),
                            "final": row["final_pos"],
                        },
                        clean_cache,
                    )
                    append_attention_rows(
                        attention_rows,
                        slot_row,
                        component,
                        spec,
                        "corrupt",
                        batch_index,
                        int(row["final_pos"]),
                        {
                            "BOS": row["bos_pos"],
                            "The": row.get("corrupt_initial_the_pos"),
                            "patient": row.get("corrupt_patient_pos"),
                            "was": row.get("corrupt_was_pos"),
                            "verb": row.get("corrupt_verb_pos"),
                            "by": row.get("corrupt_by_pos"),
                            "final": row["final_pos"],
                        },
                        corrupt_cache,
                    )
        del clean_cache, corrupt_cache

    readout_df = pd.DataFrame(readout_rows)
    activation_df = pd.DataFrame(activation_rows)
    attention_df = pd.DataFrame(attention_rows)

    if not readout_df.empty:
        readout_summary = (
            readout_df.groupby(
                SLOT_COLUMNS + ["selected_budget", "component", "component_type", "layer", "head", "position_label", "condition"],
                dropna=False,
            )
            .agg(readout_mean=("readout", "mean"), readout_std=("readout", "std"), example_count=("readout", "count"))
            .reset_index()
        )
        pivot = (
            readout_summary.pivot_table(
                index=SLOT_COLUMNS + ["selected_budget", "component", "component_type", "layer", "head", "position_label"],
                columns="condition",
                values="readout_mean",
                aggfunc="first",
            )
            .reset_index()
        )
        if {"clean", "corrupt"}.issubset(pivot.columns):
            pivot["clean_minus_corrupt_readout"] = pivot["clean"] - pivot["corrupt"]
        pivot["readout_direction"] = pivot.apply(classify_readout_row, axis=1)
        readout_summary = readout_summary.merge(
            pivot,
            on=SLOT_COLUMNS + ["selected_budget", "component", "component_type", "layer", "head", "position_label"],
            how="left",
        )
    else:
        readout_summary = pd.DataFrame()

    if not attention_df.empty:
        attention_summary = (
            attention_df.groupby(
                SLOT_COLUMNS
                + ["selected_budget", "component", "layer", "head", "condition", "query_position_label", "key_position_label"],
                dropna=False,
            )
            .agg(
                attention_mass_mean=("attention_mass", "mean"),
                attention_mass_std=("attention_mass", "std"),
                example_count=("attention_mass", "count"),
            )
            .reset_index()
        )
    else:
        attention_summary = pd.DataFrame()

    geometry_summary, pca_projection = pca_geometry_summary(activation_df) if not activation_df.empty else (pd.DataFrame(), pd.DataFrame())
    target_metadata = pd.DataFrame(
        [
            {
                "model_slug": slot_row["model_slug"],
                "run_name": slot_row["run_name"],
                "sample_size": int(slot_row["sample_size"]),
                "seed": int(slot_row["seed"]),
                "selected_budget": int(slot_row["selected_budget"]),
                "model_name": context.get("model_name"),
                "requested_model_name": context.get("requested_model_name"),
                "target_source": context.get("target_source"),
                "target_filter_policy": context.get("target_filter_policy"),
                "target_settings_source": context.get("target_settings_source"),
                "summary_path": context.get("summary_path"),
                "example_count": int(len(examples)),
            }
        ]
    )

    del context["model"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "attention_semantics": attention_summary,
        "readout_semantics": readout_summary,
        "activation_geometry_summary": geometry_summary,
        "activation_pca_projection": pca_projection,
        "target_metadata": target_metadata,
    }


def concat_result_table(results: list[dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    frames = [result[key] for result in results if key in result and isinstance(result[key], pd.DataFrame) and not result[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_report(report_dir: Path, model_slugs: list[str] | None, seeds: list[int] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(report_dir / "necessary_budget_summary.csv")
    collapsed = pd.read_csv(report_dir / "necessary_collapsed_edges.csv")
    require_columns(
        summary,
        {"model_slug", "run_name", "sample_size", "seed", "status", "edge_path", "selected_budget"},
        "necessary_budget_summary.csv",
    )
    require_columns(
        collapsed,
        {
            "model_slug",
            "run_name",
            "sample_size",
            "seed",
            "selected_budget",
            "collapsed_edge",
            "parent",
            "child",
            "signed_sum",
            "abs_score",
            "rank",
            "underlying_edge_count",
        },
        "necessary_collapsed_edges.csv",
    )
    selected = summary[summary["status"].isin(["selected", "selected_from_partial"])].copy()
    if model_slugs is not None:
        selected = selected[selected["model_slug"].isin(model_slugs)].copy()
        collapsed = collapsed[collapsed["model_slug"].isin(model_slugs)].copy()
    if seeds is not None:
        selected = selected[selected["seed"].isin(seeds)].copy()
        collapsed = collapsed[collapsed["seed"].isin(seeds)].copy()
    if selected.empty:
        raise ValueError("No selected necessary-edge slots matched the requested filters.")
    semantic_edges = add_edge_semantics(collapsed)
    component_cards = component_cards_from_edges(semantic_edges)
    return selected, semantic_edges, component_cards


def build_prompt_intersection(
    project_root: Path,
    selected_summary: pd.DataFrame,
    batch_size_override: int | None,
) -> pd.DataFrame | None:
    prompt_frames = []
    for _, slot_row in selected_summary.drop_duplicates("model_slug").iterrows():
        meta = slot_run_metadata(project_root, slot_row)
        try:
            frame = normalize_prompt_columns(
                load_policy_filtered_dataset(
                    project_root=project_root,
                    model_name=meta["model_name"],
                    batch_size=batch_size_for_model(str(slot_row["model_slug"]), batch_size_override),
                    target_filter_policy=meta["target_filter_policy"],
                    cache=True,
                    max_examples=None,
                    seed=int(slot_row["seed"]),
                    target_source=meta["target_source"],
                )
            )
            prompt_frames.append(prompt_pair_columns(frame))
        except Exception as exc:
            print(f"Could not load prompt pairs for {slot_row['model_slug']}: {exc}")
    if not prompt_frames:
        return None
    prompt_pairs = intersect_prompt_pair_frames(prompt_frames)
    print(f"Using {len(prompt_pairs)} prompt pairs in the cross-model intersection.")
    return prompt_pairs


def write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def output_tables_from_results(activation_results: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    return {
        "attention_semantics": concat_result_table(activation_results, "attention_semantics"),
        "readout_semantics": concat_result_table(activation_results, "readout_semantics"),
        "activation_geometry_summary": concat_result_table(activation_results, "activation_geometry_summary"),
        "activation_pca_projection": concat_result_table(activation_results, "activation_pca_projection"),
        "target_metadata": concat_result_table(activation_results, "target_metadata"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute activation-heavy semantic tables for necessary subcircuits.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override all per-model batch sizes. Defaults: gpt2=128, "
            "meta-llama_Llama-3.2-3B=128, google_gemma-3-4b-pt=72, Qwen_Qwen3-4B=32."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[42], help="Seeds to analyze. Use --all-seeds to disable this filter.")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--model-slug", action="append", default=None, help="Model slug to include; repeat for multiple models.")
    parser.add_argument("--no-prompt-intersection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_animacy_circuit_root(args.project_root)
    results_root = project_root / "results" / "eap_ig_localization"
    report_dir = args.report_dir if args.report_dir is not None else latest_report_dir(results_root)
    output_dir = args.output_dir or (results_root / f"necessary_semantics_activations_{date.today().isoformat()}")
    seeds = None if args.all_seeds else args.seeds

    selected_summary, semantic_edges, component_cards = load_report(report_dir, args.model_slug, seeds)
    prompt_pairs = None if args.no_prompt_intersection else build_prompt_intersection(project_root, selected_summary, args.batch_size)

    activation_results = []
    slot_batch_sizes = {}
    completed_slots = []
    base_metadata = {
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "n_examples": int(args.n_examples),
        "batch_size_override": int(args.batch_size) if args.batch_size is not None else None,
        "default_batch_size_by_model_slug": DEFAULT_BATCH_SIZE_BY_MODEL_SLUG,
        "process_order_by_model_slug": MODEL_PROCESS_ORDER,
        "seeds": seeds,
        "model_slugs": args.model_slug,
        "use_prompt_intersection": not args.no_prompt_intersection,
        "selected_slot_count": int(len(selected_summary)),
    }
    for _, slot_row in sorted_slots(selected_summary).iterrows():
        slot_batch_size = batch_size_for_model(str(slot_row["model_slug"]), args.batch_size)
        slot_key = f"{slot_row['model_slug']}|{slot_row['run_name']}|{slot_row['sample_size']}|{slot_row['seed']}"
        slot_batch_sizes[slot_key] = int(slot_batch_size)
        print(
            f"Analyzing {slot_row['model_slug']} / {slot_row['run_name']} / seed {slot_row['seed']} "
            f"with n_examples={args.n_examples}, batch_size={slot_batch_size}"
        )
        activation_results.append(
            analyze_slot_activations(
                project_root=project_root,
                semantic_edges=semantic_edges,
                component_cards=component_cards,
                slot_row=slot_row,
                n_examples=args.n_examples,
                batch_size=slot_batch_size,
                prompt_pairs=prompt_pairs,
            )
        )
        completed_slots.append(slot_key)
        tables = output_tables_from_results(activation_results)
        metadata = {
            **base_metadata,
            "slot_batch_sizes": slot_batch_sizes,
            "completed_slots": completed_slots,
            "completed_slot_count": int(len(completed_slots)),
            "table_rows": {name: int(len(frame)) for name, frame in tables.items()},
        }
        write_outputs(output_dir, tables, metadata)
        print(f"Checkpointed activation semantic tables to {output_dir}")

    tables = output_tables_from_results(activation_results)
    metadata = {
        **base_metadata,
        "slot_batch_sizes": slot_batch_sizes,
        "completed_slots": completed_slots,
        "completed_slot_count": int(len(completed_slots)),
        "table_rows": {name: int(len(frame)) for name, frame in tables.items()},
    }
    write_outputs(output_dir, tables, metadata)
    print(f"Wrote activation semantic tables to {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import gc
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from circuit_finder_paths import ensure_dir, resolve_animacy_circuit_root
from utils import canonical_model_name, date_tag, model_note, safe_model_name, save_csv, save_json, save_torch, timestamp_tag


DEFAULT_THRESHOLDS = (50, 60, 70, 80, 90)
DEFAULT_BUDGETS = (10, 25, 50, 100, 200)
DEFAULT_EAP_BUDGETS = (
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
)
DEFAULT_EAP_FIXED_BUDGET_PREFIX = (30, 40, 50, 60, 70, 80, 90, 100, 200, 300)
DEFAULT_EAP_BUDGET_MAX_FRACTION = 0.15
DEFAULT_EAP_BUDGET_FLOOR = 2000
DEFAULT_EAP_BUDGET_TAIL_POINTS = 20
DEFAULT_EAP_EARLY_STOP_THRESHOLD = 0.85
DEFAULT_EAP_EARLY_STOP_PATIENCE = 5
DEFAULT_EAP_EARLY_STOP_MIN_DELTA = 0.01
DEFAULT_EAP_EARLY_STOP_START_BUDGET = 300
DEFAULT_CONCEPT_ALPHA_GRID = (-10.0, -7.5, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)
DEFAULT_CONCEPT_HOOK_POINTS = ("hook_resid_pre", "hook_resid_mid", "hook_resid_post")
RELATIVE_BUDGET_TICKVALS = (0.005, *tuple(value / 100.0 for value in range(1, 11)))
RELATIVE_BUDGET_TICKTEXT = ("0.5%", *tuple(f"{value}%" for value in range(1, 11)))

DEFAULT_TOKENIZATION_FILTER_MODELS = (
    "gpt2",
    "meta-llama/Llama-3.2-3B",
    "google/gemma-3-4b-pt",
    "Qwen/Qwen3-4B",
)
DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS = DEFAULT_TOKENIZATION_FILTER_MODELS
TOKENIZATION_FILTER_VERSION = "component_single_token_v1"
CHOSEN_DATASET_METRIC = "avg_LD_pairs"
DEFAULT_DISCOVERY_MARGIN_THRESHOLD = 0.5
DEFAULT_TARGET_SOURCE = "wordnet"
TARGET_SOURCE_FILES = {
    "wordnet": "wordnet_lexname_targets_500x500.json",
    "abstract_agency": "abstract_agency_targets_strict_90x500.json",
}
METRIC_INVESTIGATION_METRICS = (
    "Delta_P",
    CHOSEN_DATASET_METRIC,
    "avg_LD_top_k",
    "LD_someone_something",
)

MODEL_SPECIFIC_CORRECT = "model_specific_correct"
SHARED_CORRECT = "shared_correct"
DATASET_SET_NAMES = (MODEL_SPECIFIC_CORRECT, SHARED_CORRECT)


@dataclass(frozen=True)
class PatchSite:
    layer: int
    token_position_from_end: int
    token: str
    score: float


@dataclass
class BudgetEvaluation:
    budget: int
    faithfulness_mean: float
    faithfulness_std: float
    example_count: int
    collapsed_edge_budget: int
    expanded_edge_count: int
    induced_node_count: int


@dataclass
class VariantSelectionSummary:
    variant_id: str
    pipeline: str
    threshold: int | None
    supported_budgets: list[int]
    mean_faithfulness: float
    mean_induced_nodes: float


@dataclass
class ComparisonConfig:
    model_name: str = "gpt2"
    dataset_filter_model_name: str = "gpt2"
    seed: int = 42
    validation_fraction: float = 0.2
    ig_steps: int = 5
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    max_examples: int | None = None
    metric_batch_size: int = 32
    patch_batch_size: int = 16
    attribution_batch_size: int = 8
    evaluation_batch_size: int = 1
    output_stem: str = "full_vs_hybrid_eap_ig"
    selection_tolerance: float = 1e-4
    dataset_filter_path: str | None = None
    refresh_dataset_filter: bool = False
    cache_dataset_filter: bool = True
    target_filter_policy: str = "model_success"
    target_source: str = DEFAULT_TARGET_SOURCE


@dataclass
class EAPExperimentConfig:
    model_name: str = "gpt2"
    dataset_filter_model_name: str = "gpt2"
    seed: int = 42
    discovery_sample_size: int = 500
    discovery_margin_threshold: float | None = DEFAULT_DISCOVERY_MARGIN_THRESHOLD
    filter_batch_size: int = 50
    attribution_batch_size: int = 8
    evaluation_batch_size: int = 1
    ig_steps: int = 5
    budgets: tuple[int, ...] | None = None
    budget_max_fraction: float = DEFAULT_EAP_BUDGET_MAX_FRACTION
    budget_floor: int = DEFAULT_EAP_BUDGET_FLOOR
    budget_tail_points: int = DEFAULT_EAP_BUDGET_TAIL_POINTS
    budget_early_stop: bool = False
    budget_early_stop_threshold: float = DEFAULT_EAP_EARLY_STOP_THRESHOLD
    budget_early_stop_patience: int = DEFAULT_EAP_EARLY_STOP_PATIENCE
    budget_early_stop_min_delta: float = DEFAULT_EAP_EARLY_STOP_MIN_DELTA
    budget_early_stop_start_budget: int = DEFAULT_EAP_EARLY_STOP_START_BUDGET
    output_day: str | None = None
    dataset_filter_path: str | None = None
    refresh_dataset_filter: bool = False
    cache_dataset_filter: bool = True
    max_filter_examples: int | None = None
    target_filter_policy: str = "model_success"
    target_source: str = DEFAULT_TARGET_SOURCE
    circuit_finder_day: str | None = None
    importance_quantile: float = 0.10
    component_discovery_threshold: int | None = None


@dataclass
class DualSetExperimentConfig:
    model_name: str = "gpt2"
    shared_filter_model_names: tuple[str, ...] = ("gpt2",)
    dataset_set_names: tuple[str, ...] = DATASET_SET_NAMES
    seed: int = 42
    discovery_sample_size: int = 500
    discovery_margin_threshold: float | None = DEFAULT_DISCOVERY_MARGIN_THRESHOLD
    filter_batch_size: int = 50
    attribution_batch_size: int = 8
    evaluation_batch_size: int = 1
    ig_steps: int = 5
    budgets: tuple[int, ...] | None = None
    budget_max_fraction: float = DEFAULT_EAP_BUDGET_MAX_FRACTION
    budget_floor: int = DEFAULT_EAP_BUDGET_FLOOR
    budget_tail_points: int = DEFAULT_EAP_BUDGET_TAIL_POINTS
    budget_early_stop: bool = False
    budget_early_stop_threshold: float = DEFAULT_EAP_EARLY_STOP_THRESHOLD
    budget_early_stop_patience: int = DEFAULT_EAP_EARLY_STOP_PATIENCE
    budget_early_stop_min_delta: float = DEFAULT_EAP_EARLY_STOP_MIN_DELTA
    budget_early_stop_start_budget: int = DEFAULT_EAP_EARLY_STOP_START_BUDGET
    output_day: str | None = None
    dataset_filter_path: str | None = None
    refresh_dataset_filter: bool = False
    cache_dataset_filter: bool = True
    max_filter_examples: int | None = None
    target_filter_policy: str = "model_success"
    target_source: str = DEFAULT_TARGET_SOURCE
    run_diagnose: bool = True
    run_eap: bool = True


@dataclass
class EAPShadowRediscoveryConfig:
    model_name: str = "gpt2"
    shared_filter_model_names: tuple[str, ...] = ("gpt2",)
    dataset_set_name: str = MODEL_SPECIFIC_CORRECT
    seed: int = 42
    discovery_sample_size: int = 500
    discovery_margin_threshold: float | None = DEFAULT_DISCOVERY_MARGIN_THRESHOLD
    filter_batch_size: int = 50
    attribution_batch_size: int = 8
    evaluation_batch_size: int = 1
    ig_steps: int = 5
    budgets: tuple[int, ...] | None = None
    budget_max_fraction: float = DEFAULT_EAP_BUDGET_MAX_FRACTION
    budget_floor: int = DEFAULT_EAP_BUDGET_FLOOR
    budget_tail_points: int = DEFAULT_EAP_BUDGET_TAIL_POINTS
    budget_early_stop: bool = False
    budget_early_stop_threshold: float = DEFAULT_EAP_EARLY_STOP_THRESHOLD
    budget_early_stop_patience: int = DEFAULT_EAP_EARLY_STOP_PATIENCE
    budget_early_stop_min_delta: float = DEFAULT_EAP_EARLY_STOP_MIN_DELTA
    budget_early_stop_start_budget: int = DEFAULT_EAP_EARLY_STOP_START_BUDGET
    output_day: str | None = None
    dataset_filter_path: str | None = None
    refresh_dataset_filter: bool = False
    cache_dataset_filter: bool = True
    max_filter_examples: int | None = None
    target_filter_policy: str = "model_success"
    target_source: str = DEFAULT_TARGET_SOURCE
    main_experiment_path: str | None = None
    source_faithfulness_threshold: float = DEFAULT_EAP_EARLY_STOP_THRESHOLD
    variants: tuple[str, ...] | None = None
    top_edge_count: int = 100
    skip_existing: bool = False


@dataclass
class ComponentDiscoveryConfig:
    model_name: str = "gpt2"
    dataset_filter_model_name: str = "gpt2"
    seed: int = 42
    discovery_sample_size: int = 500
    discovery_margin_threshold: float | None = DEFAULT_DISCOVERY_MARGIN_THRESHOLD
    filter_batch_size: int = 50
    patch_batch_size: int = 16
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
    output_day: str | None = None
    dataset_filter_path: str | None = None
    refresh_dataset_filter: bool = False
    cache_dataset_filter: bool = True
    max_filter_examples: int | None = None
    target_filter_policy: str = "model_success"
    target_source: str = DEFAULT_TARGET_SOURCE


@dataclass
class ConceptExtractionConfig:
    model_name: str = "gpt2"
    dataset_filter_model_name: str = "gpt2"
    seed: int = 42
    filter_batch_size: int = 50
    extraction_batch_size: int = 16
    steering_batch_size: int = 16
    alpha_grid: tuple[float, ...] = DEFAULT_CONCEPT_ALPHA_GRID
    hook_points: tuple[str, ...] = DEFAULT_CONCEPT_HOOK_POINTS
    normalize_concept_vector: bool = True
    selection_effect_fraction: float = 0.90
    random_control_repeats: int = 10
    output_day: str | None = None
    dataset_filter_path: str | None = None
    refresh_dataset_filter: bool = False
    cache_dataset_filter: bool = True
    max_filter_examples: int | None = None
    max_concept_examples: int | None = None
    target_filter_policy: str = "model_success"
    target_source: str = DEFAULT_TARGET_SOURCE


class CircuitPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.clean = df["clean_prefix"].tolist()
        self.corrupt = df["corrupt_prefix"].tolist()
        self.corrupt_metric = torch.tensor(
            df["corrupt_metric"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int) -> tuple[str, str, torch.Tensor]:
        return self.clean[idx], self.corrupt[idx], self.corrupt_metric[idx]


def results_root(project_root: Path) -> Path:
    return ensure_dir(project_root / "results")


def grouped_model_results_root(project_root: Path, group_name: str, model_name: str) -> Path:
    return ensure_dir(
        results_root(project_root)
        / group_name
        / safe_model_name(canonical_model_name(model_name))
    )


def grouped_model_day_dir(
    project_root: Path,
    group_name: str,
    model_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(
        grouped_model_results_root(project_root, group_name, model_name)
        / (date_tag() if day is None else day)
    )


def grouped_model_dataset_set_day_dir(
    project_root: Path,
    group_name: str,
    model_name: str,
    dataset_set_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(
        grouped_model_results_root(project_root, group_name, model_name)
        / dataset_set_name
        / (date_tag() if day is None else day)
    )


def model_diagnostic_dir(project_root: Path, model_name: str, day: str | None = None) -> Path:
    return grouped_model_day_dir(project_root, "model_diagnostic", model_name, day)


def model_diagnostic_dataset_set_dir(
    project_root: Path,
    model_name: str,
    dataset_set_name: str,
    day: str | None = None,
) -> Path:
    return grouped_model_dataset_set_day_dir(
        project_root,
        "model_diagnostic",
        model_name,
        dataset_set_name,
        day,
    )


def manual_circuit_discovery_dir(
    project_root: Path,
    model_name: str,
    day: str | None = None,
) -> Path:
    return grouped_model_day_dir(project_root, "manual_circuit_discovery", model_name, day)


def manual_circuit_checkpoints_dir(
    project_root: Path,
    model_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(manual_circuit_discovery_dir(project_root, model_name, day) / "checkpoints")


def manual_circuit_components_dir(
    project_root: Path,
    model_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(manual_circuit_discovery_dir(project_root, model_name, day) / "components")


def manual_circuit_plots_dir(
    project_root: Path,
    model_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(manual_circuit_image_dir(project_root, model_name, day) / "plots")


def eap_ig_dir(project_root: Path, model_name: str, day: str | None = None) -> Path:
    return grouped_model_day_dir(project_root, "eap_ig", model_name, day)


def eap_ig_dataset_set_dir(
    project_root: Path,
    model_name: str,
    dataset_set_name: str,
    day: str | None = None,
) -> Path:
    return grouped_model_dataset_set_day_dir(
        project_root,
        "eap_ig",
        model_name,
        dataset_set_name,
        day,
    )


def eap_ig_full_model_dir(project_root: Path, model_name: str, day: str | None = None) -> Path:
    return ensure_dir(eap_ig_dir(project_root, model_name, day) / "full_model")


def eap_ig_dataset_set_full_model_dir(
    project_root: Path,
    model_name: str,
    dataset_set_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(eap_ig_dataset_set_dir(project_root, model_name, dataset_set_name, day) / "full_model")


def eap_ig_dataset_set_shadow_rediscovery_dir(
    project_root: Path,
    model_name: str,
    dataset_set_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(
        eap_ig_dataset_set_dir(project_root, model_name, dataset_set_name, day)
        / "shadow_rediscovery"
    )


def eap_ig_selected_components_dir(
    project_root: Path,
    model_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(eap_ig_dir(project_root, model_name, day) / "selected_components")


def eap_ig_comparison_dir(project_root: Path, model_name: str, day: str | None = None) -> Path:
    return ensure_dir(eap_ig_dir(project_root, model_name, day) / "comparison")


def concept_extraction_dir(project_root: Path, model_name: str, day: str | None = None) -> Path:
    return grouped_model_day_dir(project_root, "concept_extraction", model_name, day)


def image_root(project_root: Path) -> Path:
    return ensure_dir(results_root(project_root) / "images")


def grouped_model_images_root(project_root: Path, group_name: str, model_name: str) -> Path:
    return ensure_dir(
        image_root(project_root)
        / group_name
        / safe_model_name(canonical_model_name(model_name))
    )


def grouped_model_day_image_dir(
    project_root: Path,
    group_name: str,
    model_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(
        grouped_model_images_root(project_root, group_name, model_name)
        / (date_tag() if day is None else day)
    )


def grouped_model_dataset_set_day_image_dir(
    project_root: Path,
    group_name: str,
    model_name: str,
    dataset_set_name: str,
    day: str | None = None,
) -> Path:
    return ensure_dir(
        grouped_model_images_root(project_root, group_name, model_name)
        / dataset_set_name
        / (date_tag() if day is None else day)
    )


def model_diagnostic_image_dir(
    project_root: Path,
    model_name: str,
    day: str | None = None,
) -> Path:
    return grouped_model_day_image_dir(project_root, "model_diagnostic", model_name, day)


def model_diagnostic_dataset_set_image_dir(
    project_root: Path,
    model_name: str,
    dataset_set_name: str,
    day: str | None = None,
) -> Path:
    return grouped_model_dataset_set_day_image_dir(
        project_root,
        "model_diagnostic",
        model_name,
        dataset_set_name,
        day,
    )


def manual_circuit_image_dir(
    project_root: Path,
    model_name: str,
    day: str | None = None,
) -> Path:
    return grouped_model_day_image_dir(project_root, "manual_circuit_discovery", model_name, day)


def eap_ig_image_dir(project_root: Path, model_name: str, day: str | None = None) -> Path:
    return grouped_model_day_image_dir(project_root, "eap_ig", model_name, day)


def eap_ig_dataset_set_image_dir(
    project_root: Path,
    model_name: str,
    dataset_set_name: str,
    day: str | None = None,
) -> Path:
    return grouped_model_dataset_set_day_image_dir(
        project_root,
        "eap_ig",
        model_name,
        dataset_set_name,
        day,
    )


def resolve_image_output_dir(project_root: Path, file_path: Path) -> Path:
    try:
        relative = file_path.resolve().relative_to(results_root(project_root).resolve())
    except ValueError:
        return ensure_dir(image_root(project_root) / "legacy" / "global")

    parts = relative.parts
    if len(parts) >= 3 and parts[0] == "model_diagnostic":
        destination = image_root(project_root) / "model_diagnostic" / parts[1]
        if len(parts) >= 4 and parts[2] in DATASET_SET_NAMES:
            destination = destination / parts[2] / parts[3]
            if len(parts) >= 6:
                destination /= parts[4]
        else:
            destination /= parts[2]
            if len(parts) >= 5:
                destination /= parts[3]
        return ensure_dir(destination)

    if len(parts) >= 4 and parts[0] == "manual_circuit_discovery":
        destination = image_root(project_root) / "manual_circuit_discovery" / parts[1] / parts[2]
        if parts[3] in {"checkpoints", "components", "plots"}:
            destination /= parts[3]
        return ensure_dir(destination)

    if len(parts) >= 4 and parts[0] == "eap_ig":
        destination = image_root(project_root) / "eap_ig" / parts[1]
        part_index = 2
        if parts[2] in DATASET_SET_NAMES:
            destination = destination / parts[2] / parts[3]
            part_index = 4
        else:
            destination /= parts[2]
            part_index = 3
        if len(parts) > part_index and parts[part_index] in {
            "full_model",
            "selected_components",
            "comparison",
            "shadow_rediscovery",
        }:
            destination /= parts[part_index]
            if parts[part_index] == "shadow_rediscovery" and len(parts) > part_index + 1:
                destination /= parts[part_index + 1]
        return ensure_dir(destination)

    return ensure_dir(image_root(project_root) / "legacy" / "global")


def experiment_output_dir(
    project_root: Path,
    stem: str,
    model_name: str = "gpt2",
    day: str | None = None,
) -> Path:
    return ensure_dir(eap_ig_comparison_dir(project_root, model_name, day) / stem)


def generated_fallback_root(project_root: Path) -> Path:
    base = Path("C:/tmp") if Path("C:/tmp").drive else Path("/tmp")
    return ensure_dir(base / "grammatical-circuits" / project_root.name)


def ensure_generated_dir(primary: Path, project_root: Path, *fallback_parts: str) -> Path:
    try:
        return ensure_dir(primary)
    except PermissionError:
        return ensure_dir(generated_fallback_root(project_root).joinpath(*fallback_parts))


def load_animacy_dataframe(project_root: Path) -> pd.DataFrame:
    dataset_path = project_root / "dataset" / "semantic_meaningful" / "filtered_single_token_pairs.jsonl"
    dataset = pd.read_json(dataset_path, lines=True)[["clean", "corrupt"]].rename(
        columns={"clean": "clean_prefix", "corrupt": "corrupt_prefix"}
    )
    return dataset.drop_duplicates(subset=["clean_prefix", "corrupt_prefix"]).reset_index(drop=True)


def metric_filtered_dataset_path(project_root: Path) -> Path:
    return (
        project_root
        / "dataset"
        / "semantic_meaningful"
        / "metric_filtered"
        / "mp_filtered_avg_LD_pairs.csv"
    )


def metric_filtered_model_dataset_dir(project_root: Path, model_name: str) -> Path:
    return ensure_dir(
        project_root
        / "dataset"
        / "semantic_meaningful"
        / "metric_filtered"
        / safe_model_name(canonical_model_name(model_name))
    )


def metric_filtered_model_dataset_path(
    project_root: Path,
    model_name: str,
    metric_name: str = CHOSEN_DATASET_METRIC,
) -> Path:
    return metric_filtered_model_dataset_dir(project_root, model_name) / f"mp_filtered_{metric_name}.csv"


def metric_scored_model_dataset_path(project_root: Path, model_name: str) -> Path:
    model_slug = safe_model_name(canonical_model_name(model_name))
    return metric_filtered_model_dataset_dir(project_root, model_name) / f"mp_scored_metrics_{model_slug}.csv"


def find_metric_filtered_model_dataset_path(
    project_root: Path,
    model_name: str,
    metric_name: str = CHOSEN_DATASET_METRIC,
) -> Path | None:
    model_path = metric_filtered_model_dataset_path(project_root, model_name, metric_name)
    if model_path.is_file():
        return model_path
    if canonical_model_name(model_name) == "gpt2":
        legacy_path = metric_filtered_dataset_path(project_root)
        if legacy_path.is_file():
            return legacy_path
    return None


def normalize_metric_dataset_columns(
    dataset: pd.DataFrame,
    metric_name: str = CHOSEN_DATASET_METRIC,
) -> pd.DataFrame:
    normalized = dataset.copy()
    rename_map = {}
    if "clean_prefix" not in normalized.columns and "clean" in normalized.columns:
        rename_map["clean"] = "clean_prefix"
    if "corrupt_prefix" not in normalized.columns and "corrupt" in normalized.columns:
        rename_map["corrupt"] = "corrupt_prefix"
    if "clean_metric" not in normalized.columns and f"clean_{metric_name}" in normalized.columns:
        rename_map[f"clean_{metric_name}"] = "clean_metric"
    if "corrupt_metric" not in normalized.columns and f"corrupt_{metric_name}" in normalized.columns:
        rename_map[f"corrupt_{metric_name}"] = "corrupt_metric"
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    required_columns = ["clean_prefix", "corrupt_prefix", "clean_metric", "corrupt_metric"]
    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError(
            f"The metric-filtered {metric_name} dataset is missing required columns: "
            f"{missing}"
        )
    return normalized


def load_metric_filtered_model_success_dataset(
    project_root: Path,
    model_name: str,
    metric_name: str = CHOSEN_DATASET_METRIC,
    path: Path | None = None,
    common_filter_model_names: Sequence[str] = DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS,
) -> pd.DataFrame:
    resolved_model_name = canonical_model_name(model_name)
    scored_path = metric_scored_model_dataset_path(project_root, resolved_model_name)
    dataset_path = (
        path
        or (scored_path if scored_path.is_file() else None)
        or find_metric_filtered_model_dataset_path(project_root, resolved_model_name, metric_name)
    )
    if dataset_path is None:
        raise FileNotFoundError(
            f"No metric-filtered {metric_name} dataset found for {resolved_model_name}."
        )

    dataset = normalize_metric_dataset_columns(pd.read_csv(dataset_path), metric_name)
    success_df = attach_pair_metadata(dataset.copy(), project_root)
    common_pairs = load_common_tokenized_pairs(project_root, common_filter_model_names)
    success_df = filter_df_to_prompt_pairs(success_df, prompt_pair_columns(common_pairs))
    success_df = filter_model_success_examples(success_df)
    success_df = success_df.assign(
        filter_model_name=resolved_model_name,
        filter_clean_metric=success_df["clean_metric"],
        filter_corrupt_metric=success_df["corrupt_metric"],
        filter_metric_name=metric_name,
        filter_source_path=str(dataset_path),
    )
    normalized = normalize_model_success_metadata(success_df, resolved_model_name)
    normalized.attrs["model_success_cache_path"] = str(dataset_path)
    normalized.attrs["model_success_cache_status"] = "loaded_metric_filtered"
    return normalized


def load_metric_filtered_gpt2_success_dataset(project_root: Path) -> pd.DataFrame:
    return load_metric_filtered_model_success_dataset(project_root, "gpt2")


def load_metric_scored_model_dataset(
    project_root: Path,
    model_name: str,
    metric_name: str = CHOSEN_DATASET_METRIC,
    common_filter_model_names: Sequence[str] = DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS,
) -> pd.DataFrame | None:
    path = metric_scored_model_dataset_path(project_root, model_name)
    if not path.is_file():
        return None
    dataset = normalize_metric_dataset_columns(pd.read_csv(path), metric_name)
    dataset = attach_pair_metadata(dataset, project_root)
    common_pairs = load_common_tokenized_pairs(project_root, common_filter_model_names)
    dataset = filter_df_to_prompt_pairs(dataset, prompt_pair_columns(common_pairs))
    dataset = dataset.assign(
        filter_model_name=canonical_model_name(model_name),
        filter_clean_metric=dataset["clean_metric"],
        filter_corrupt_metric=dataset["corrupt_metric"],
        filter_metric_name=metric_name,
        filter_source_path=str(path),
    )
    return normalize_model_success_metadata(dataset, canonical_model_name(model_name))


def load_common_tokenized_pairs(
    project_root: Path,
    model_names: Sequence[str] = DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS,
) -> pd.DataFrame:
    resolved_model_names = unique_model_names(model_names)
    path = tokenization_filter_intersection_pairs_path(project_root, resolved_model_names)
    if not path.is_file():
        artifact = prepare_tokenization_filter_artifacts(
            project_root,
            resolved_model_names,
            refresh=False,
        )
        path = Path(artifact["paths"]["intersection_pairs"])
    return pd.read_csv(path)


def load_tokenization_filtered_pairs_for_model(project_root: Path, model_name: str) -> pd.DataFrame | None:
    return load_common_tokenized_pairs(project_root)


def load_animacy_pair_metadata(project_root: Path) -> pd.DataFrame:
    metadata_path = project_root / "dataset" / "semantic_meaningful" / "raw_pairs_semantic.jsonl"
    if not metadata_path.is_file():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "clean_prefix": item.get("clean"),
                    "corrupt_prefix": item.get("corrupt"),
                    "patient": item.get("patient"),
                    "clean_verb": item.get("clean_verb"),
                    "corrupt_verb": item.get("corrupt_verb"),
                    "uid": item.get("uid"),
                    "domain": item.get("domain"),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["clean_prefix", "corrupt_prefix"]
    ).reset_index(drop=True)


def attach_pair_metadata(df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    metadata = load_animacy_pair_metadata(project_root)
    if metadata.empty:
        return df.copy()
    return df.merge(metadata, on=["clean_prefix", "corrupt_prefix"], how="left")


def resolve_target_source_path(project_root: Path, target_source: str | Path | None = None) -> Path:
    source = str(target_source or DEFAULT_TARGET_SOURCE)
    if source in TARGET_SOURCE_FILES:
        return project_root / "dataset" / "semantic_meaningful" / TARGET_SOURCE_FILES[source]

    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate


def target_source_slug(project_root: Path, target_source: str | Path | None = None) -> str:
    source = str(target_source or DEFAULT_TARGET_SOURCE)
    if source in TARGET_SOURCE_FILES:
        return source
    path = resolve_target_source_path(project_root, source)
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"custom_{path.stem}_{digest}"


def target_source_label(project_root: Path, target_source: str | Path | None = None) -> str:
    slug = target_source_slug(project_root, target_source)
    return "" if slug == DEFAULT_TARGET_SOURCE else f"{safe_model_name(slug)}_"


def load_animacy_targets(
    project_root: Path,
    target_source: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    targets_path = resolve_target_source_path(project_root, target_source)
    blacklist_path = project_root / "dataset" / "semantic_meaningful" / "blacklist.json"

    with targets_path.open("r", encoding="utf-8") as handle:
        targets_data = json.load(handle)

    targets_raw = pd.json_normalize(targets_data["targets"])
    blacklist = json.loads(blacklist_path.read_text(encoding="utf-8")) if blacklist_path.is_file() else {}
    use_blacklist = target_source_slug(project_root, target_source) == DEFAULT_TARGET_SOURCE

    blocked = {
        "animate": (
            set(blacklist.get("animate", {}).get("remove_now", []))
            | set(blacklist.get("animate", {}).get("strongly_consider_removing", []))
        )
        if use_blacklist
        else set(),
        "inanimate": (
            set(blacklist.get("inanimate", {}).get("remove_now", []))
            | set(blacklist.get("inanimate", {}).get("strongly_consider_removing", []))
        )
        if use_blacklist
        else set(),
    }

    animate_words = [
        word
        for word in targets_raw.loc[0, "animate"]
        if word not in blocked["animate"]
    ]
    inanimate_words = [
        word
        for word in targets_raw.loc[0, "inanimate"]
        if word not in blocked["inanimate"]
    ]
    return animate_words, inanimate_words


def tokenization_filtered_dataset_dir(project_root: Path) -> Path:
    return ensure_dir(
        project_root / "dataset" / "semantic_meaningful" / "tokenization_filtered"
    )


def tokenization_filter_pairs_path(project_root: Path, model_name: str) -> Path:
    return (
        tokenization_filtered_dataset_dir(project_root)
        / f"pairs_{safe_model_name(canonical_model_name(model_name))}.csv"
    )


def tokenization_filter_jsonl_pairs_path(project_root: Path, model_name: str) -> Path:
    return (
        tokenization_filtered_dataset_dir(project_root)
        / f"accepted_pairs_{safe_model_name(canonical_model_name(model_name))}.jsonl"
    )


def tokenization_filter_intersection_pairs_path(
    project_root: Path,
    model_names: Sequence[str],
) -> Path:
    resolved_names = sorted(unique_model_names(model_names))
    digest = hashlib.sha256("\n".join(resolved_names).encode("utf-8")).hexdigest()[:12]
    return tokenization_filtered_dataset_dir(project_root) / f"pairs_intersection_{digest}.csv"


def tokenization_filter_intersection_jsonl_pairs_path(
    project_root: Path,
    model_names: Sequence[str],
) -> Path:
    resolved_names = sorted(unique_model_names(model_names))
    digest = hashlib.sha256("\n".join(resolved_names).encode("utf-8")).hexdigest()[:12]
    return tokenization_filtered_dataset_dir(project_root) / f"accepted_pairs_intersection_{digest}.jsonl"


def tokenization_filter_targets_path(project_root: Path, model_names: Sequence[str]) -> Path:
    return tokenization_filter_targets_path_for_source(
        project_root,
        model_names,
        target_source=DEFAULT_TARGET_SOURCE,
    )


def tokenization_filter_targets_path_for_source(
    project_root: Path,
    model_names: Sequence[str],
    target_source: str | Path | None = None,
) -> Path:
    resolved_names = unique_model_names(model_names)
    if len(resolved_names) == 1:
        stem = safe_model_name(resolved_names[0])
    else:
        resolved_names = sorted(resolved_names)
        digest = hashlib.sha256("\n".join(resolved_names).encode("utf-8")).hexdigest()[:12]
        stem = f"intersection_{digest}"
    stem = f"{target_source_label(project_root, target_source)}{stem}"
    return tokenization_filtered_dataset_dir(project_root) / f"targets_{stem}.json"


def tokenization_filter_summary_path(project_root: Path) -> Path:
    return tokenization_filtered_dataset_dir(project_root) / "summary.json"


def tokenizer_input_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    return list(input_ids)


def is_single_token_target(word: str, tokenizer) -> bool:
    return len(tokenizer_input_ids(tokenizer, " " + word)) == 1


def target_filter_diagnostics(
    animate_words: Sequence[str],
    inanimate_words: Sequence[str],
    filtered_animate_words: Sequence[str],
    filtered_inanimate_words: Sequence[str],
    filter_model_names: Sequence[str],
) -> dict[str, Any]:
    animate_set = set(filtered_animate_words)
    inanimate_set = set(filtered_inanimate_words)
    return {
        "filter_version": TOKENIZATION_FILTER_VERSION,
        "filter_model_names": list(filter_model_names),
        "animate": {
            "original_count": int(len(animate_words)),
            "filtered_count": int(len(filtered_animate_words)),
            "dropped_count": int(len(animate_words) - len(filtered_animate_words)),
            "retention_rate": (
                float(len(filtered_animate_words) / len(animate_words))
                if animate_words
                else 0.0
            ),
            "dropped_examples": [word for word in animate_words if word not in animate_set][:20],
        },
        "inanimate": {
            "original_count": int(len(inanimate_words)),
            "filtered_count": int(len(filtered_inanimate_words)),
            "dropped_count": int(len(inanimate_words) - len(filtered_inanimate_words)),
            "retention_rate": (
                float(len(filtered_inanimate_words) / len(inanimate_words))
                if inanimate_words
                else 0.0
            ),
            "dropped_examples": [word for word in inanimate_words if word not in inanimate_set][:20],
        },
    }


def filter_targets_for_models(
    project_root: Path,
    model_names: Sequence[str],
    *,
    target_tokenizer=None,
    target_source: str | Path | None = None,
) -> dict[str, Any]:
    animate_words, inanimate_words = load_animacy_targets(project_root, target_source=target_source)
    filter_model_names = unique_model_names(model_names)
    if not filter_model_names:
        raise ValueError("At least one model name is required to filter targets.")

    tokenizer_by_model: dict[str, Any] = {}
    for model_name in filter_model_names:
        if (
            target_tokenizer is not None
            and len(filter_model_names) == 1
            and model_name == filter_model_names[0]
        ):
            tokenizer_by_model[model_name] = target_tokenizer
        else:
            tokenizer_by_model[model_name] = load_hf_tokenizer(model_name)

    filtered_animate = [
        word
        for word in animate_words
        if all(is_single_token_target(word, tokenizer) for tokenizer in tokenizer_by_model.values())
    ]
    filtered_inanimate = [
        word
        for word in inanimate_words
        if all(is_single_token_target(word, tokenizer) for tokenizer in tokenizer_by_model.values())
    ]

    return {
        "animate": filtered_animate,
        "inanimate": filtered_inanimate,
        "summary": target_filter_diagnostics(
            animate_words,
            inanimate_words,
            filtered_animate,
            filtered_inanimate,
            filter_model_names,
        ),
    }


def save_filtered_targets(
    project_root: Path,
    model_names: Sequence[str],
    payload: dict[str, Any],
    *,
    target_source: str | Path | None = None,
) -> Path:
    path = tokenization_filter_targets_path_for_source(
        project_root,
        model_names,
        target_source=target_source,
    )
    resolved_names = unique_model_names(model_names)
    if len(resolved_names) > 1:
        resolved_names = sorted(resolved_names)
    save_json(
        path,
        {
            "filter_version": TOKENIZATION_FILTER_VERSION,
            "target_source": str(target_source or DEFAULT_TARGET_SOURCE),
            "target_source_path": str(resolve_target_source_path(project_root, target_source)),
            "model_names": resolved_names,
            "targets": {
                "animate": payload["animate"],
                "inanimate": payload["inanimate"],
            },
            "summary": payload["summary"],
        },
    )
    return path


def load_filtered_targets_from_file(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload["targets"]
    return (
        list(targets["animate"]),
        list(targets["inanimate"]),
        payload.get("summary", {}),
    )


def load_or_filter_targets_for_models(
    project_root: Path,
    model_names: Sequence[str],
    *,
    target_tokenizer=None,
    refresh: bool = False,
    target_source: str | Path | None = None,
) -> tuple[list[str], list[str], dict[str, Any], Path | None]:
    path = tokenization_filter_targets_path_for_source(
        project_root,
        model_names,
        target_source=target_source,
    )
    if path.is_file() and not refresh:
        animate_words, inanimate_words, summary = load_filtered_targets_from_file(path)
        return animate_words, inanimate_words, summary, path

    payload = filter_targets_for_models(
        project_root,
        model_names,
        target_tokenizer=target_tokenizer,
        target_source=target_source,
    )
    path = save_filtered_targets(project_root, model_names, payload, target_source=target_source)
    return payload["animate"], payload["inanimate"], payload["summary"], path


def load_model(model_name: str) -> HookedTransformer:
    from transformer_lens import HookedTransformer
    from transformer_lens.model_bridge import TransformerBridge

    resolved_model_name = canonical_model_name(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = HookedTransformer.from_pretrained(resolved_model_name, device=device)
    except Exception as hooked_exc:
        try:
            model = TransformerBridge.boot_transformers(resolved_model_name, device=device)
            model.enable_compatibility_mode(disable_warnings=True)
        except Exception as bridge_exc:
            note = model_note(resolved_model_name)
            detail = f" {note}" if note else ""
            raise RuntimeError(
                "Could not load "
                f"{model_name!r} as {resolved_model_name!r} with TransformerLens "
                "HookedTransformer or TransformerBridge."
                f"{detail}"
            ) from bridge_exc
        else:
            model._eap_bridge_fallback_from = hooked_exc
    if hasattr(model.cfg, "default_prepend_bos"):
        model.cfg.default_prepend_bos = True
    elif isinstance(model.cfg, dict):
        model.cfg["default_prepend_bos"] = True
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    if hasattr(model, "set_use_split_qkv_input"):
        model.set_use_split_qkv_input(True)
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    if hasattr(model.cfg, "use_attn_result"):
        model.cfg.use_attn_result = True
    if hasattr(model.cfg, "use_split_qkv_input"):
        model.cfg.use_split_qkv_input = True
    if hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True
    if getattr(model.cfg, "n_key_value_heads", None) is not None:
        model.cfg.ungroup_grouped_query_attention = True
    return model


def load_hf_tokenizer(model_name: str):
    from transformers import AutoProcessor, AutoTokenizer

    resolved_model_name = canonical_model_name(model_name)
    offline_mode = str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or str(os.environ.get("TRANSFORMERS_OFFLINE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    tokenizer_errors: list[Exception] = []
    tokenizer_attempts = [
        {"local_files_only": True, "extra_special_tokens": {}},
        {"local_files_only": True, "use_fast": False, "extra_special_tokens": {}},
        {"local_files_only": True, "trust_remote_code": True, "extra_special_tokens": {}},
        {
            "local_files_only": True,
            "trust_remote_code": True,
            "use_fast": False,
            "extra_special_tokens": {},
        },
    ]
    if not offline_mode:
        tokenizer_attempts.extend(
            [
                {"extra_special_tokens": {}},
                {"use_fast": False, "extra_special_tokens": {}},
                {"trust_remote_code": True, "extra_special_tokens": {}},
                {
                    "trust_remote_code": True,
                    "use_fast": False,
                    "extra_special_tokens": {},
                },
            ]
        )
    for kwargs in tokenizer_attempts:
        try:
            return AutoTokenizer.from_pretrained(resolved_model_name, **kwargs)
        except Exception as exc:
            tokenizer_errors.append(exc)

    processor_errors: list[Exception] = []
    processor_attempts = [
        {"local_files_only": True, "trust_remote_code": True},
        {"local_files_only": True},
    ]
    if not offline_mode:
        processor_attempts.extend(
            [
                {"trust_remote_code": True},
                {},
            ]
        )
    for kwargs in processor_attempts:
        try:
            processor = AutoProcessor.from_pretrained(resolved_model_name, **kwargs)
        except Exception as exc:
            processor_errors.append(exc)
            continue
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            processor_errors.append(
                RuntimeError(
                    f"Loaded a processor for {model_name!r} as {resolved_model_name!r}, "
                    "but it does not expose a text tokenizer."
                )
            )
            continue
        return tokenizer

    try:
        return load_tokenizers_json_tokenizer(
            resolved_model_name,
            local_files_only=offline_mode,
        )
    except Exception as raw_tokenizer_error:
        last_error = raw_tokenizer_error
        if processor_errors:
            last_error = processor_errors[-1]
        elif tokenizer_errors:
            last_error = tokenizer_errors[-1]
        raise RuntimeError(
            f"Could not load a tokenizer or processor for {model_name!r} "
            f"as {resolved_model_name!r}. Last error: {raw_tokenizer_error}"
        ) from last_error


class TokenizersJsonWrapper:
    def __init__(self, tokenizer, name_or_path: str):
        self.tokenizer = tokenizer
        self.name_or_path = name_or_path

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_: Any,
    ):
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        payload: dict[str, Any] = {"input_ids": encoded.ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = encoded.offsets
        return SimpleNamespace(**payload)

    def decode(self, token_ids: Sequence[int] | int, **_: Any) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(list(token_ids))


def load_tokenizers_json_tokenizer(
    model_name: str,
    *,
    local_files_only: bool = True,
) -> TokenizersJsonWrapper:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer_path = hf_hub_download(
        model_name,
        "tokenizer.json",
        local_files_only=local_files_only,
    )
    return TokenizersJsonWrapper(
        Tokenizer.from_file(tokenizer_path),
        name_or_path=model_name,
    )


def create_target_tensor(
    word_list: Sequence[str],
    tokenizer,
    device: str | torch.device,
) -> torch.Tensor:
    valid_ids: list[int] = []
    for word in word_list:
        tokens = tokenizer(" " + word, add_special_tokens=False).input_ids
        if len(tokens) != 1:
            print(
                f"WARNING: The word '{word}' split into {len(tokens)} tokens {tokens}. Skipping."
            )
            continue
        valid_ids.append(tokens[0])
    if not valid_ids:
        raise ValueError(
            "None of the requested target words are single tokens for this model/tokenizer."
        )
    return torch.tensor(valid_ids, dtype=torch.long, device=device)


def verify_target_tensors(
    original_list: Sequence[str],
    ids_tensor: torch.Tensor,
    tokenizer,
) -> None:
    decoded_words = [tokenizer.decode([token_id]) for token_id in ids_tensor]
    decoded_clean = [word.strip() for word in decoded_words]
    original_clean = [word.strip() for word in original_list]
    for word in decoded_clean:
        assert (
            word in original_clean
        ), f"Tokenizer mapped an ID to '{word}', which is not in the target set."


def target_tokenization_summary(
    word_list: Sequence[str],
    tokenizer,
    max_examples: int = 20,
) -> dict[str, Any]:
    multi_token_examples: list[dict[str, Any]] = []
    empty_token_examples: list[str] = []
    single_token_count = 0
    multi_token_count = 0
    empty_token_count = 0

    for word in word_list:
        token_ids = tokenizer(" " + word, add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            single_token_count += 1
        elif len(token_ids) == 0:
            empty_token_count += 1
            if len(empty_token_examples) < max_examples:
                empty_token_examples.append(word)
        else:
            multi_token_count += 1
            if len(multi_token_examples) < max_examples:
                multi_token_examples.append(
                    {
                        "word": word,
                        "token_count": len(token_ids),
                        "token_ids": token_ids,
                        "tokens": [tokenizer.decode([token_id]) for token_id in token_ids],
                    }
                )

    total = len(word_list)
    return {
        "total": total,
        "single_token_count": single_token_count,
        "single_token_rate": float(single_token_count / total) if total else 0.0,
        "multi_token_count": multi_token_count,
        "empty_token_count": empty_token_count,
        "multi_token_examples": multi_token_examples,
        "empty_token_examples": empty_token_examples,
    }


def build_target_tokenization_diagnostics(
    animate_words: Sequence[str],
    inanimate_words: Sequence[str],
    tokenizer,
) -> dict[str, Any]:
    return {
        "animate": target_tokenization_summary(animate_words, tokenizer),
        "inanimate": target_tokenization_summary(inanimate_words, tokenizer),
    }


def get_input_ids_with_bos(
    text_or_texts,
    model: HookedTransformer,
):
    tokens = model.to_tokens(
        text_or_texts,
        prepend_bos=True,
        move_to_device=False,
    )
    return tokens[0] if isinstance(text_or_texts, str) else tokens


def find_word_char_span(text: str, word: str) -> tuple[int, int] | None:
    pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)")
    match = pattern.search(text)
    if match is None:
        return None
    return match.start(), match.end()


def tokenizer_offsets(tokenizer, text: str) -> list[tuple[int, int]] | None:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        return None

    offsets = getattr(encoded, "offset_mapping", None)
    if offsets is None and isinstance(encoded, dict):
        offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    return [(int(start), int(end)) for start, end in offsets]


def token_span_from_offsets(
    offsets: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    overlapping = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not overlapping:
        return None
    return min(overlapping), max(overlapping) + 1


def token_count_no_special(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def component_token_span(
    tokenizer,
    text: str,
    component: str,
) -> tuple[tuple[int, int] | None, str | None]:
    char_span = find_word_char_span(text, component)
    if char_span is None:
        return None, "component_not_found"

    offsets = tokenizer_offsets(tokenizer, text)
    if offsets is not None:
        token_span = token_span_from_offsets(offsets, *char_span)
        if token_span is not None:
            return token_span, None

    start, end = char_span
    prefix_token_count = token_count_no_special(tokenizer, text[:start])
    prefix_component_token_count = token_count_no_special(tokenizer, text[:end])
    if prefix_component_token_count <= prefix_token_count:
        return None, "component_token_span_empty"
    return (prefix_token_count, prefix_component_token_count), None


def add_count(summary: dict[str, int], key: str) -> None:
    summary[key] = int(summary.get(key, 0) + 1)


def add_limited_example(
    examples: list[dict[str, Any]],
    item: dict[str, Any],
    max_examples: int,
) -> None:
    if len(examples) < max_examples:
        examples.append(item)


def pair_token_alignment_details(
    row: pd.Series,
    tokenizer,
    *,
    metadata_available: bool,
) -> dict[str, Any]:
    clean_text = str(row["clean_prefix"])
    corrupt_text = str(row["corrupt_prefix"])
    details: dict[str, Any] = {
        "clean_prefix": clean_text,
        "corrupt_prefix": corrupt_text,
        "pair_ok": True,
        "clean_len": token_count_no_special(tokenizer, clean_text),
        "corrupt_len": token_count_no_special(tokenizer, corrupt_text),
        "patient": None,
        "clean_verb": None,
        "corrupt_verb": None,
        "clean_patient_span": None,
        "corrupt_patient_span": None,
        "clean_verb_span": None,
        "corrupt_verb_span": None,
        "clean_patient_error": None,
        "corrupt_patient_error": None,
        "clean_verb_error": None,
        "corrupt_verb_error": None,
    }
    if details["clean_len"] != details["corrupt_len"]:
        details["pair_ok"] = False

    if not metadata_available:
        details["pair_ok"] = False
        details["metadata_missing"] = True
        return details

    patient = row["patient"]
    clean_verb = row["clean_verb"]
    corrupt_verb = row["corrupt_verb"]
    if any(pd.isna(value) for value in (patient, clean_verb, corrupt_verb)):
        details["pair_ok"] = False
        details["metadata_missing"] = True
        return details

    details["patient"] = str(patient)
    details["clean_verb"] = str(clean_verb)
    details["corrupt_verb"] = str(corrupt_verb)
    details["metadata_missing"] = False

    details["clean_patient_span"], details["clean_patient_error"] = component_token_span(
        tokenizer,
        clean_text,
        details["patient"],
    )
    details["corrupt_patient_span"], details["corrupt_patient_error"] = component_token_span(
        tokenizer,
        corrupt_text,
        details["patient"],
    )
    details["clean_verb_span"], details["clean_verb_error"] = component_token_span(
        tokenizer,
        clean_text,
        details["clean_verb"],
    )
    details["corrupt_verb_span"], details["corrupt_verb_error"] = component_token_span(
        tokenizer,
        corrupt_text,
        details["corrupt_verb"],
    )

    if (
        details["clean_patient_span"] is None
        or details["corrupt_patient_span"] is None
        or details["clean_patient_span"] != details["corrupt_patient_span"]
        or details["clean_verb_span"] is None
        or details["corrupt_verb_span"] is None
        or details["clean_verb_span"] != details["corrupt_verb_span"]
    ):
        details["pair_ok"] = False

    if details["clean_patient_span"] is not None:
        details["clean_patient_token_width"] = (
            details["clean_patient_span"][1] - details["clean_patient_span"][0]
        )
        if details["clean_patient_token_width"] != 1:
            details["pair_ok"] = False
    if details["corrupt_patient_span"] is not None:
        details["corrupt_patient_token_width"] = (
            details["corrupt_patient_span"][1] - details["corrupt_patient_span"][0]
        )
        if details["corrupt_patient_token_width"] != 1:
            details["pair_ok"] = False
    if details["clean_verb_span"] is not None:
        details["clean_verb_token_width"] = (
            details["clean_verb_span"][1] - details["clean_verb_span"][0]
        )
        if details["clean_verb_token_width"] != 1:
            details["pair_ok"] = False
    if details["corrupt_verb_span"] is not None:
        details["corrupt_verb_token_width"] = (
            details["corrupt_verb_span"][1] - details["corrupt_verb_span"][0]
        )
        if details["corrupt_verb_token_width"] != 1:
            details["pair_ok"] = False

    return details


def filter_token_aligned_pairs(
    df: pd.DataFrame,
    tokenizer,
) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True).copy()

    metadata_available = {"patient", "clean_verb", "corrupt_verb"}.issubset(df.columns)
    kept_indices = [
        idx
        for idx, row in df.iterrows()
        if pair_token_alignment_details(
            row,
            tokenizer,
            metadata_available=metadata_available,
        )["pair_ok"]
    ]
    return df.loc[kept_indices].reset_index(drop=True).copy()


def token_alignment_diagnostics(
    df: pd.DataFrame,
    tokenizer,
    max_examples: int = 20,
) -> dict[str, Any]:
    required_metadata = {"patient", "clean_verb", "corrupt_verb"}
    metadata_available = required_metadata.issubset(df.columns)
    counts: dict[str, int] = {
        "total_pairs": int(len(df)),
        "checked_pairs": 0,
        "metadata_missing": 0,
        "sequence_length_mismatch": 0,
        "patient_span_failure": 0,
        "verb_span_failure": 0,
        "patient_span_misaligned": 0,
        "verb_span_misaligned": 0,
        "patient_not_single_token": 0,
        "verb_not_single_token": 0,
        "patient_token_width_mismatch": 0,
        "verb_token_width_mismatch": 0,
        "fully_aligned": 0,
    }
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row_idx, row in df.reset_index(drop=True).iterrows():
        details = pair_token_alignment_details(
            row,
            tokenizer,
            metadata_available=metadata_available,
        )
        if details["metadata_missing"]:
            counts["metadata_missing"] += 1
            add_limited_example(
                examples["metadata_missing"],
                {
                    "row": int(row_idx),
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                },
                max_examples,
            )
            continue

        counts["checked_pairs"] += 1
        pair_ok = True
        if details["clean_len"] != details["corrupt_len"]:
            pair_ok = False
            add_count(counts, "sequence_length_mismatch")
            add_limited_example(
                examples["sequence_length_mismatch"],
                {
                    "row": int(row_idx),
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                    "clean_token_count": details["clean_len"],
                    "corrupt_token_count": details["corrupt_len"],
                },
                max_examples,
            )

        if details["clean_patient_span"] is None or details["corrupt_patient_span"] is None:
            pair_ok = False
            add_count(counts, "patient_span_failure")
            add_limited_example(
                examples["patient_span_failure"],
                {
                    "row": int(row_idx),
                    "patient": details["patient"],
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                    "clean_error": details["clean_patient_error"],
                    "corrupt_error": details["corrupt_patient_error"],
                },
                max_examples,
            )
        elif details["clean_patient_span"] != details["corrupt_patient_span"]:
            pair_ok = False
            add_count(counts, "patient_span_misaligned")
            add_limited_example(
                examples["patient_span_misaligned"],
                {
                    "row": int(row_idx),
                    "patient": details["patient"],
                    "clean_span": details["clean_patient_span"],
                    "corrupt_span": details["corrupt_patient_span"],
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                },
                max_examples,
            )

        if details["clean_verb_span"] is None or details["corrupt_verb_span"] is None:
            pair_ok = False
            add_count(counts, "verb_span_failure")
            add_limited_example(
                examples["verb_span_failure"],
                {
                    "row": int(row_idx),
                    "clean_verb": details["clean_verb"],
                    "corrupt_verb": details["corrupt_verb"],
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                    "clean_error": details["clean_verb_error"],
                    "corrupt_error": details["corrupt_verb_error"],
                },
                max_examples,
            )
        elif details["clean_verb_span"] != details["corrupt_verb_span"]:
            pair_ok = False
            add_count(counts, "verb_span_misaligned")
            clean_width = details["clean_verb_span"][1] - details["clean_verb_span"][0]
            corrupt_width = details["corrupt_verb_span"][1] - details["corrupt_verb_span"][0]
            if clean_width != corrupt_width:
                add_count(counts, "verb_token_width_mismatch")
            add_limited_example(
                examples["verb_span_misaligned"],
                {
                    "row": int(row_idx),
                    "clean_verb": details["clean_verb"],
                    "corrupt_verb": details["corrupt_verb"],
                    "clean_span": details["clean_verb_span"],
                    "corrupt_span": details["corrupt_verb_span"],
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                },
                max_examples,
            )

        if details["clean_patient_span"] is not None and details["corrupt_patient_span"] is not None:
            clean_patient_width = details["clean_patient_span"][1] - details["clean_patient_span"][0]
            corrupt_patient_width = details["corrupt_patient_span"][1] - details["corrupt_patient_span"][0]
            if clean_patient_width != corrupt_patient_width:
                pair_ok = False
                add_count(counts, "patient_token_width_mismatch")
            if clean_patient_width != 1 or corrupt_patient_width != 1:
                pair_ok = False
                add_count(counts, "patient_not_single_token")
                add_limited_example(
                    examples["patient_not_single_token"],
                    {
                        "row": int(row_idx),
                        "patient": details["patient"],
                        "clean_width": int(clean_patient_width),
                        "corrupt_width": int(corrupt_patient_width),
                        "clean_prefix": details["clean_prefix"],
                        "corrupt_prefix": details["corrupt_prefix"],
                    },
                    max_examples,
                )

        if details["clean_verb_span"] is not None and details["corrupt_verb_span"] is not None:
            clean_verb_width = details["clean_verb_span"][1] - details["clean_verb_span"][0]
            corrupt_verb_width = details["corrupt_verb_span"][1] - details["corrupt_verb_span"][0]
            if clean_verb_width != 1 or corrupt_verb_width != 1:
                pair_ok = False
                add_count(counts, "verb_not_single_token")
                add_limited_example(
                    examples["verb_not_single_token"],
                    {
                        "row": int(row_idx),
                        "clean_verb": details["clean_verb"],
                        "corrupt_verb": details["corrupt_verb"],
                        "clean_width": int(clean_verb_width),
                        "corrupt_width": int(corrupt_verb_width),
                        "clean_prefix": details["clean_prefix"],
                        "corrupt_prefix": details["corrupt_prefix"],
                    },
                    max_examples,
                )

        if pair_ok:
            counts["fully_aligned"] += 1

    checked = counts["checked_pairs"]
    return {
        **counts,
        "fully_aligned_rate_among_checked": (
            float(counts["fully_aligned"] / checked) if checked else 0.0
        ),
        "examples": dict(examples),
    }


def add_sequence_lengths(df: pd.DataFrame, model: HookedTransformer) -> pd.DataFrame:
    valid_indices: list[int] = []
    for idx, row in df.iterrows():
        clean_tokens = get_input_ids_with_bos(row["clean_prefix"], model)
        corrupt_tokens = get_input_ids_with_bos(row["corrupt_prefix"], model)
        if len(clean_tokens) == len(corrupt_tokens):
            valid_indices.append(idx)

    aligned = df.loc[valid_indices].reset_index(drop=True).copy()
    aligned["seq_len"] = aligned["clean_prefix"].apply(
        lambda text: len(get_input_ids_with_bos(text, model))
    )
    return aligned


def maybe_limit_examples(df: pd.DataFrame, max_examples: int | None, seed: int) -> pd.DataFrame:
    if max_examples is None or max_examples >= len(df):
        return df.reset_index(drop=True).copy()
    return (
        df.sample(n=max_examples, random_state=seed)
        .reset_index(drop=True)
        .copy()
    )


def generate_exact_length_batches(
    df: pd.DataFrame,
    model: HookedTransformer,
    batch_size: int,
    device: str | torch.device,
):
    grouped = df.groupby("seq_len")
    for length, group in grouped:
        for start in range(0, len(group), batch_size):
            batch_df = group.iloc[start : start + batch_size]
            clean_tokens = get_input_ids_with_bos(
                batch_df["clean_prefix"].tolist(),
                model,
            ).to(device)
            corrupt_tokens = get_input_ids_with_bos(
                batch_df["corrupt_prefix"].tolist(),
                model,
            ).to(device)
            assert clean_tokens.shape == corrupt_tokens.shape
            assert clean_tokens.shape[1] == length
            yield clean_tokens, corrupt_tokens, batch_df


def average_logit_difference_from_final_logits(
    final_logits: torch.Tensor,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
) -> torch.Tensor:
    anim_logits = final_logits[:, animate_ids_tensor]
    inan_logits = final_logits[:, inanimate_ids_tensor]
    return anim_logits.mean(dim=-1) - inan_logits.mean(dim=-1)


def final_token_average_logit_difference(
    logits: torch.Tensor,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
) -> torch.Tensor:
    assert logits.ndim == 3, f"Expected [batch, pos, vocab], got {tuple(logits.shape)}"
    logit_diff = average_logit_difference_from_final_logits(
        logits[:, -1, :],
        animate_ids_tensor,
        inanimate_ids_tensor,
    )
    assert not torch.isnan(logit_diff).any(), "NaNs generated during logit difference calculation."
    return logit_diff


def compute_sequence_metrics(
    df: pd.DataFrame,
    model: HookedTransformer,
    tokenizer,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    batch_size: int,
) -> pd.DataFrame:
    clean_metric_map: dict[int, float] = {}
    corrupt_metric_map: dict[int, float] = {}

    estimated_batches = sum(math.ceil(len(group) / batch_size) for _, group in df.groupby("seq_len"))
    batch_generator = generate_exact_length_batches(
        df=df,
        model=model,
        batch_size=batch_size,
        device=model.cfg.device,
    )

    for clean_tokens, corrupt_tokens, batch_df in tqdm(
        batch_generator,
        total=estimated_batches,
        desc="Scoring clean/corrupt pairs",
    ):
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)

        clean_metric = average_logit_difference_from_final_logits(
            clean_logits[:, -1, :],
            animate_ids_tensor,
            inanimate_ids_tensor,
        )
        corrupt_metric = average_logit_difference_from_final_logits(
            corrupt_logits[:, -1, :],
            animate_ids_tensor,
            inanimate_ids_tensor,
        )

        for index, clean_value, corrupt_value in zip(
            batch_df.index.tolist(),
            clean_metric.cpu().tolist(),
            corrupt_metric.cpu().tolist(),
        ):
            clean_metric_map[index] = float(clean_value)
            corrupt_metric_map[index] = float(corrupt_value)

    scored = df.copy()
    scored["clean_metric"] = [clean_metric_map[idx] for idx in scored.index]
    scored["corrupt_metric"] = [corrupt_metric_map[idx] for idx in scored.index]
    return scored


def probability_difference_from_final_probs(
    final_probs: torch.Tensor,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
) -> torch.Tensor:
    anim_probs = final_probs[:, animate_ids_tensor]
    inan_probs = final_probs[:, inanimate_ids_tensor]
    return anim_probs.mean(dim=-1) - inan_probs.mean(dim=-1)


def top_k_logit_difference_from_final_logits(
    final_logits: torch.Tensor,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    k: int,
) -> torch.Tensor:
    effective_k = min(
        int(k),
        int(animate_ids_tensor.numel()),
        int(inanimate_ids_tensor.numel()),
    )
    if effective_k <= 0:
        return torch.full(
            (final_logits.shape[0],),
            float("nan"),
            dtype=final_logits.dtype,
            device=final_logits.device,
        )
    anim_logits = final_logits[:, animate_ids_tensor]
    inan_logits = final_logits[:, inanimate_ids_tensor]
    anim_top = torch.topk(anim_logits, effective_k, dim=-1).values
    inan_top = torch.topk(inan_logits, effective_k, dim=-1).values
    return anim_top.mean(dim=-1) - inan_top.mean(dim=-1)


def single_token_id_or_none(tokenizer, text: str) -> int | None:
    token_ids = tokenizer_input_ids(tokenizer, text)
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def someone_vs_something_from_prev_logits(
    prev_logits: torch.Tensor,
    someone_id: int | None,
    something_id: int | None,
) -> torch.Tensor:
    if someone_id is None or something_id is None:
        return torch.full(
            (prev_logits.shape[0],),
            float("nan"),
            dtype=prev_logits.dtype,
            device=prev_logits.device,
        )
    return prev_logits[:, someone_id] - prev_logits[:, something_id]


def compute_metric_investigation_scores(
    df: pd.DataFrame,
    model: HookedTransformer,
    tokenizer,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    batch_size: int,
    top_k: int = 50,
) -> pd.DataFrame:
    working = df.copy()
    if "clean" not in working.columns and "clean_prefix" in working.columns:
        working["clean"] = working["clean_prefix"]
    if "corrupt" not in working.columns and "corrupt_prefix" in working.columns:
        working["corrupt"] = working["corrupt_prefix"]
    if "clean_prefix" not in working.columns:
        working["clean_prefix"] = working["clean"]
    if "corrupt_prefix" not in working.columns:
        working["corrupt_prefix"] = working["corrupt"]

    aligned_df = add_sequence_lengths(working, model)
    animate_ids_tensor = animate_ids_tensor.to(model.cfg.device)
    inanimate_ids_tensor = inanimate_ids_tensor.to(model.cfg.device)
    someone_id = single_token_id_or_none(tokenizer, " someone")
    something_id = single_token_id_or_none(tokenizer, " something")

    clean_values: dict[str, dict[int, float]] = {
        metric: {} for metric in METRIC_INVESTIGATION_METRICS
    }
    corrupt_values: dict[str, dict[int, float]] = {
        metric: {} for metric in METRIC_INVESTIGATION_METRICS
    }

    estimated_batches = sum(
        math.ceil(len(group) / batch_size)
        for _, group in aligned_df.groupby("seq_len")
    )
    batch_generator = generate_exact_length_batches(
        df=aligned_df,
        model=model,
        batch_size=batch_size,
        device=model.cfg.device,
    )

    for clean_tokens, corrupt_tokens, batch_df in tqdm(
        batch_generator,
        total=estimated_batches,
        desc="Scoring metric-investigation pairs",
    ):
        with torch.no_grad():
            clean_logits_all = model(clean_tokens)
            corrupt_logits_all = model(corrupt_tokens)

        clean_final_logits = clean_logits_all[:, -1, :]
        corrupt_final_logits = corrupt_logits_all[:, -1, :]
        clean_final_probs = clean_final_logits.softmax(dim=-1)
        corrupt_final_probs = corrupt_final_logits.softmax(dim=-1)
        clean_prev_logits = clean_logits_all[:, -2, :]
        corrupt_prev_logits = corrupt_logits_all[:, -2, :]

        batch_metric_values = {
            "clean": {
                "Delta_P": probability_difference_from_final_probs(
                    clean_final_probs,
                    animate_ids_tensor,
                    inanimate_ids_tensor,
                ),
                "avg_LD_pairs": average_logit_difference_from_final_logits(
                    clean_final_logits,
                    animate_ids_tensor,
                    inanimate_ids_tensor,
                ),
                "avg_LD_top_k": top_k_logit_difference_from_final_logits(
                    clean_final_logits,
                    animate_ids_tensor,
                    inanimate_ids_tensor,
                    top_k,
                ),
                "LD_someone_something": someone_vs_something_from_prev_logits(
                    clean_prev_logits,
                    someone_id,
                    something_id,
                ),
            },
            "corrupt": {
                "Delta_P": probability_difference_from_final_probs(
                    corrupt_final_probs,
                    animate_ids_tensor,
                    inanimate_ids_tensor,
                ),
                "avg_LD_pairs": average_logit_difference_from_final_logits(
                    corrupt_final_logits,
                    animate_ids_tensor,
                    inanimate_ids_tensor,
                ),
                "avg_LD_top_k": top_k_logit_difference_from_final_logits(
                    corrupt_final_logits,
                    animate_ids_tensor,
                    inanimate_ids_tensor,
                    top_k,
                ),
                "LD_someone_something": someone_vs_something_from_prev_logits(
                    corrupt_prev_logits,
                    someone_id,
                    something_id,
                ),
            },
        }

        for position, index in enumerate(batch_df.index.tolist()):
            for metric in METRIC_INVESTIGATION_METRICS:
                clean_values[metric][index] = float(
                    batch_metric_values["clean"][metric][position].detach().cpu()
                )
                corrupt_values[metric][index] = float(
                    batch_metric_values["corrupt"][metric][position].detach().cpu()
                )

        del clean_logits_all, corrupt_logits_all

    scored = aligned_df.copy()
    for metric in METRIC_INVESTIGATION_METRICS:
        scored[f"clean_{metric}"] = [
            clean_values[metric][idx] for idx in scored.index
        ]
        scored[f"corrupt_{metric}"] = [
            corrupt_values[metric][idx] for idx in scored.index
        ]
    scored["clean_metric"] = scored["clean_avg_LD_pairs"]
    scored["corrupt_metric"] = scored["corrupt_avg_LD_pairs"]
    return scored.reset_index(drop=True).copy()


def filter_model_success_examples(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["clean_metric"] > 0) & (df["corrupt_metric"] < 0)
    return df.loc[mask].reset_index(drop=True).copy()


def filter_discovery_margin_candidates(
    df: pd.DataFrame,
    margin_threshold: float | None = DEFAULT_DISCOVERY_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    if margin_threshold is None:
        return df.copy()
    margin = df["clean_metric"] - df["corrupt_metric"]
    return df.loc[margin > margin_threshold].copy()


def task_accuracy_summary(df: pd.DataFrame, eps: float = 1e-6) -> dict[str, Any]:
    total = int(len(df))

    def count_and_rate(mask: pd.Series) -> dict[str, float | int]:
        count = int(mask.sum())
        return {
            "count": count,
            "rate": float(count / total) if total else 0.0,
        }

    clean_success = df["clean_metric"] > 0
    corrupt_success = df["corrupt_metric"] < 0
    pair_success = clean_success & corrupt_success
    recovery_margin = (df["clean_metric"] - df["corrupt_metric"]) > eps

    return {
        "example_count": total,
        "clean_success": count_and_rate(clean_success),
        "corrupt_success": count_and_rate(corrupt_success),
        "pair_success": count_and_rate(pair_success),
        "recovery_margin": count_and_rate(recovery_margin),
        "clean_metric_mean": float(df["clean_metric"].mean()) if total else 0.0,
        "corrupt_metric_mean": float(df["corrupt_metric"].mean()) if total else 0.0,
        "margin_mean": float((df["clean_metric"] - df["corrupt_metric"]).mean()) if total else 0.0,
    }


def filter_recovery_margin_examples(
    df: pd.DataFrame,
    eps: float = 1e-6,
) -> pd.DataFrame:
    mask = (df["clean_metric"] - df["corrupt_metric"]) > eps
    return df.loc[mask].reset_index(drop=True).copy()


def apply_target_filter_policy(
    df: pd.DataFrame,
    policy: str,
    eps: float = 1e-6,
) -> pd.DataFrame:
    if policy == "none":
        return df.reset_index(drop=True).copy()
    if policy == "recovery_margin":
        return filter_recovery_margin_examples(df, eps=eps)
    if policy == "model_success":
        return filter_model_success_examples(df)
    raise ValueError(
        "target_filter_policy must be one of: none, recovery_margin, model_success."
    )


def model_success_dataset_dir(project_root: Path) -> Path:
    return ensure_dir(results_root(project_root) / "model_success")


def model_success_cache_candidates(
    project_root: Path,
    model_name: str,
    target_source: str | Path | None = None,
) -> list[Path]:
    filename = (
        f"{target_source_label(project_root, target_source)}"
        f"{safe_model_name(canonical_model_name(model_name))}_tokenized_success.csv"
    )
    candidates = [
        project_root / "results" / "model_success" / filename,
        project_root / "results" / "shared" / "model_success" / filename,
        project_root / "dataset" / "semantic_meaningful" / "model_success" / filename,
    ]
    candidates.extend((project_root / "results").glob(f"**/{filename}"))
    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique[str(candidate.resolve())] = candidate
    return list(unique.values())


def find_model_success_dataset_path(
    project_root: Path,
    model_name: str,
    target_source: str | Path | None = None,
) -> Path | None:
    existing = [
        path
        for path in model_success_cache_candidates(project_root, model_name, target_source)
        if path.is_file()
    ]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def model_success_dataset_path(
    project_root: Path,
    model_name: str,
    target_source: str | Path | None = None,
) -> Path:
    return (
        model_success_dataset_dir(project_root)
        / (
            f"{target_source_label(project_root, target_source)}"
            f"{safe_model_name(canonical_model_name(model_name))}_tokenized_success.csv"
        )
    )


def normalize_model_success_metadata(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    resolved_model_name = canonical_model_name(model_name)
    normalized = df.copy()
    if "filter_model_name" not in normalized.columns:
        normalized["filter_model_name"] = resolved_model_name
    if "filter_clean_metric" not in normalized.columns and "clean_metric" in normalized.columns:
        normalized["filter_clean_metric"] = normalized["clean_metric"]
    if "filter_corrupt_metric" not in normalized.columns and "corrupt_metric" in normalized.columns:
        normalized["filter_corrupt_metric"] = normalized["corrupt_metric"]
    if "filter_seq_len" not in normalized.columns and "seq_len" in normalized.columns:
        normalized["filter_seq_len"] = normalized["seq_len"]
    normalized["filter_model_name"] = normalized["filter_model_name"].fillna(resolved_model_name)
    return normalized


def prompt_pair_signature(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{clean} || {corrupt}"
            for clean, corrupt in zip(df["clean_prefix"], df["corrupt_prefix"])
        ).encode("utf-8")
    ).hexdigest()


def compute_model_scored_dataset(
    project_root: Path,
    model: HookedTransformer,
    tokenizer,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    batch_size: int,
    source_df: pd.DataFrame | None = None,
    max_examples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    raw_df = load_animacy_dataframe(project_root) if source_df is None else source_df.copy()
    raw_df = attach_pair_metadata(raw_df, project_root)
    raw_df = filter_token_aligned_pairs(raw_df, tokenizer)
    raw_df = raw_df.drop_duplicates(subset=["clean_prefix", "corrupt_prefix"]).reset_index(drop=True)
    aligned_df = add_sequence_lengths(raw_df, model)
    aligned_df = maybe_limit_examples(aligned_df, max_examples, seed)
    return compute_sequence_metrics(
        aligned_df,
        model=model,
        tokenizer=tokenizer,
        animate_ids_tensor=animate_ids_tensor,
        inanimate_ids_tensor=inanimate_ids_tensor,
        batch_size=batch_size,
    )


def load_or_create_model_success_dataset(
    project_root: Path,
    model_name: str,
    batch_size: int,
    cache_path: Path | str | None = None,
    refresh: bool = False,
    cache: bool = True,
    max_examples: int | None = None,
    seed: int = 42,
    target_source: str | Path | None = None,
) -> pd.DataFrame:
    resolved_model_name = canonical_model_name(model_name)
    source_slug = target_source_slug(project_root, target_source)
    path = None
    explicit_cache_path = cache_path is not None
    use_default_cache = cache and max_examples is None
    if explicit_cache_path:
        path = Path(cache_path)
    elif use_default_cache:
        metric_filtered_path = None
        if source_slug == DEFAULT_TARGET_SOURCE:
            scored_metric_path = metric_scored_model_dataset_path(project_root, resolved_model_name)
            metric_filtered_path = (
                scored_metric_path
                if scored_metric_path.is_file()
                else find_metric_filtered_model_dataset_path(
                    project_root,
                    resolved_model_name,
                    CHOSEN_DATASET_METRIC,
                )
            )
        if metric_filtered_path is not None and not refresh:
            print(
                f"Loading {resolved_model_name} {CHOSEN_DATASET_METRIC}-filtered "
                f"dataset from {metric_filtered_path}"
            )
            metric_filtered_df = load_metric_filtered_model_success_dataset(
                project_root,
                resolved_model_name,
                CHOSEN_DATASET_METRIC,
                path=metric_filtered_path,
            )
            cache_destination = model_success_dataset_path(
                project_root,
                resolved_model_name,
                target_source=target_source,
            )
            ensure_dir(cache_destination.parent)
            save_csv(metric_filtered_df, cache_destination, index=False)
            metric_filtered_df.attrs["model_success_cache_path"] = str(metric_filtered_path)
            metric_filtered_df.attrs["model_success_cache_status"] = "loaded_metric_filtered"
            return metric_filtered_df

        path = find_model_success_dataset_path(
            project_root,
            resolved_model_name,
            target_source=target_source,
        )
        if path is None:
            path = model_success_dataset_path(
                project_root,
                resolved_model_name,
                target_source=target_source,
            )

    if path is not None and path.is_file() and not refresh:
        print(f"Loading {resolved_model_name} source-success cache from {path}")
        cached = normalize_model_success_metadata(pd.read_csv(path), resolved_model_name)
        cached.attrs["model_success_cache_path"] = str(path)
        cached.attrs["model_success_cache_status"] = "loaded"
        return cached

    metric_filtered_path = None
    if source_slug == DEFAULT_TARGET_SOURCE:
        scored_metric_path = metric_scored_model_dataset_path(project_root, resolved_model_name)
        metric_filtered_path = (
            scored_metric_path
            if scored_metric_path.is_file()
            else find_metric_filtered_model_dataset_path(
                project_root,
                resolved_model_name,
                CHOSEN_DATASET_METRIC,
            )
        )
    if metric_filtered_path is not None and not refresh:
        print(
            f"Copying {resolved_model_name} {CHOSEN_DATASET_METRIC}-filtered "
            f"dataset from {metric_filtered_path}"
        )
        source_df = load_metric_filtered_model_success_dataset(
            project_root,
            resolved_model_name,
            CHOSEN_DATASET_METRIC,
            path=metric_filtered_path,
        )
        source_df = maybe_limit_examples(source_df, max_examples, seed)
        if path is not None:
            ensure_dir(path.parent)
            save_csv(source_df, path, index=False)
            source_df.attrs["model_success_cache_path"] = str(path)
            source_df.attrs["model_success_cache_status"] = "copied_from_metric_filtered"
        return source_df

    if resolved_model_name == "gpt2" and source_slug == DEFAULT_TARGET_SOURCE:
        source_df = load_metric_filtered_gpt2_success_dataset(project_root)
        source_df = maybe_limit_examples(source_df, max_examples, seed)
        if path is not None:
            print(
                "Copying existing metric-filtered GPT-2 success dataset "
                f"to {path}"
            )
            ensure_dir(path.parent)
            save_csv(source_df, path, index=False)
            source_df.attrs["model_success_cache_path"] = str(path)
            source_df.attrs["model_success_cache_status"] = "copied_from_metric_filtered"
        else:
            print(
                "Using existing metric-filtered GPT-2 success dataset in memory "
                "without writing a cache."
            )
            source_df.attrs["model_success_cache_path"] = None
            source_df.attrs["model_success_cache_status"] = "metric_filtered_in_memory"
        return source_df

    if path is not None:
        print(
            f"No {resolved_model_name} source-success cache found; "
            f"scoring and saving to {path}"
        )
    else:
        print(
            f"Scoring {resolved_model_name} source-success pool in memory "
            "without writing a cache."
        )

    model = load_model(resolved_model_name)
    tokenizer = model.tokenizer
    animate_words, inanimate_words = load_animacy_targets(
        project_root,
        target_source=target_source,
    )
    animate_ids_tensor = create_target_tensor(animate_words, tokenizer, model.cfg.device)
    inanimate_ids_tensor = create_target_tensor(inanimate_words, tokenizer, model.cfg.device)
    verify_target_tensors(animate_words, animate_ids_tensor, tokenizer)
    verify_target_tensors(inanimate_words, inanimate_ids_tensor, tokenizer)

    scored_df = compute_model_scored_dataset(
        project_root=project_root,
        model=model,
        tokenizer=tokenizer,
        animate_ids_tensor=animate_ids_tensor,
        inanimate_ids_tensor=inanimate_ids_tensor,
        batch_size=batch_size,
        max_examples=max_examples,
        seed=seed,
    )
    success_df = filter_model_success_examples(scored_df)
    success_df = success_df.assign(
        filter_model_name=resolved_model_name,
        filter_clean_metric=success_df["clean_metric"],
        filter_corrupt_metric=success_df["corrupt_metric"],
        filter_seq_len=success_df["seq_len"],
    )

    if path is not None:
        ensure_dir(path.parent)
        save_csv(success_df, path, index=False)
        success_df.attrs["model_success_cache_path"] = str(path)
        success_df.attrs["model_success_cache_status"] = "saved"
    else:
        success_df.attrs["model_success_cache_path"] = None
        success_df.attrs["model_success_cache_status"] = "disabled"

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    normalized = normalize_model_success_metadata(success_df, resolved_model_name)
    normalized.attrs.update(success_df.attrs)
    return normalized


def load_policy_filtered_dataset(
    project_root: Path,
    model_name: str,
    batch_size: int,
    target_filter_policy: str,
    cache_path: Path | str | None = None,
    refresh: bool = False,
    cache: bool = True,
    max_examples: int | None = None,
    seed: int = 42,
    target_source: str | Path | None = None,
) -> pd.DataFrame:
    resolved_model_name = canonical_model_name(model_name)
    if target_filter_policy == "model_success":
        return load_or_create_model_success_dataset(
            project_root=project_root,
            model_name=resolved_model_name,
            batch_size=batch_size,
            cache_path=cache_path,
            refresh=refresh,
            cache=cache,
            max_examples=max_examples,
            seed=seed,
            target_source=target_source,
        )

    print(
        f"Scoring {resolved_model_name} for {target_filter_policy} filtering "
        "without using the model_success cache."
    )
    context = load_model_context(
        project_root,
        resolved_model_name,
        target_source=target_source,
    )
    scored_df = compute_model_scored_dataset(
        project_root=project_root,
        model=context["model"],
        tokenizer=context["tokenizer"],
        animate_ids_tensor=context["animate_ids_tensor"],
        inanimate_ids_tensor=context["inanimate_ids_tensor"],
        batch_size=batch_size,
        max_examples=max_examples,
        seed=seed,
    )
    filtered_df = apply_target_filter_policy(scored_df, target_filter_policy)
    filtered_df = filtered_df.assign(
        filter_model_name=resolved_model_name,
        filter_clean_metric=filtered_df["clean_metric"],
        filter_corrupt_metric=filtered_df["corrupt_metric"],
        filter_seq_len=filtered_df["seq_len"],
    )
    filtered_df.attrs["model_success_cache_path"] = None
    filtered_df.attrs["model_success_cache_status"] = "disabled_for_policy"

    del context["model"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    normalized = normalize_model_success_metadata(filtered_df, resolved_model_name)
    normalized.attrs.update(filtered_df.attrs)
    return normalized


def prompt_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty and not {"clean_prefix", "corrupt_prefix"}.issubset(df.columns):
        return pd.DataFrame(columns=["clean_prefix", "corrupt_prefix"])
    if "clean_prefix" not in df.columns and "clean" in df.columns:
        df = df.assign(clean_prefix=df["clean"])
    if "corrupt_prefix" not in df.columns and "corrupt" in df.columns:
        df = df.assign(corrupt_prefix=df["corrupt"])
    return df[["clean_prefix", "corrupt_prefix"]].drop_duplicates().reset_index(drop=True)


def filter_df_to_prompt_pairs(df: pd.DataFrame, prompt_pairs: pd.DataFrame) -> pd.DataFrame:
    if prompt_pairs.empty:
        return df.iloc[0:0].copy()
    return df.merge(prompt_pairs, on=["clean_prefix", "corrupt_prefix"], how="inner")


def unique_model_names(model_names: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for model_name in model_names:
        resolved_model_name = canonical_model_name(model_name)
        if resolved_model_name in seen:
            continue
        seen.add(resolved_model_name)
        unique.append(resolved_model_name)
    return unique


def intersect_prompt_pair_frames(prompt_pair_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not prompt_pair_frames:
        return pd.DataFrame(columns=["clean_prefix", "corrupt_prefix"])
    intersection = prompt_pair_columns(prompt_pair_frames[0])
    for frame in prompt_pair_frames[1:]:
        intersection = intersection.merge(
            prompt_pair_columns(frame),
            on=["clean_prefix", "corrupt_prefix"],
            how="inner",
        )
    return intersection.reset_index(drop=True)


def load_accepted_minimal_pair_dataframe(
    project_root: Path,
    path: Path | None = None,
) -> pd.DataFrame:
    if path is None:
        path = project_root / "dataset" / "semantic_meaningful" / "accepted_filtered_pairs.jsonl"
    if not path.is_file():
        return pd.DataFrame(columns=["clean_prefix", "corrupt_prefix"])

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["clean_prefix", "corrupt_prefix"])

    df = pd.DataFrame(rows)
    if "clean_prefix" not in df.columns and "clean" in df.columns:
        df["clean_prefix"] = df["clean"]
    if "corrupt_prefix" not in df.columns and "corrupt" in df.columns:
        df["corrupt_prefix"] = df["corrupt"]
    return df


def write_accepted_minimal_pair_jsonl(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    drop_columns = {"clean_prefix", "corrupt_prefix"}
    with path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict("records"):
            item = {key: value for key, value in row.items() if key not in drop_columns}
            if "clean" not in item and "clean_prefix" in row:
                item["clean"] = row["clean_prefix"]
            if "corrupt" not in item and "corrupt_prefix" in row:
                item["corrupt"] = row["corrupt_prefix"]
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_tokenization_filter_artifacts(
    project_root: Path,
    model_names: Sequence[str] = DEFAULT_TOKENIZATION_FILTER_MODELS,
    *,
    refresh: bool = False,
    target_source: str | Path | None = None,
) -> dict[str, Any]:
    resolved_model_names = unique_model_names(model_names)
    if not resolved_model_names:
        raise ValueError("At least one model name is required.")

    base_df = attach_pair_metadata(load_animacy_dataframe(project_root), project_root)
    accepted_df = load_accepted_minimal_pair_dataframe(project_root)
    if not accepted_df.empty:
        accepted_df = attach_pair_metadata(accepted_df, project_root)
    model_artifacts: dict[str, Any] = {}
    filtered_pair_frames: list[pd.DataFrame] = []
    filtered_accepted_frames: list[pd.DataFrame] = []

    for model_name in resolved_model_names:
        tokenizer = load_hf_tokenizer(model_name)
        pair_path = tokenization_filter_pairs_path(project_root, model_name)
        jsonl_pair_path = tokenization_filter_jsonl_pairs_path(project_root, model_name)
        target_path = tokenization_filter_targets_path_for_source(
            project_root,
            [model_name],
            target_source=target_source,
        )

        if pair_path.is_file() and not refresh:
            filtered_pairs = pd.read_csv(pair_path)
            pair_diagnostics = token_alignment_diagnostics(
                filtered_pairs,
                tokenizer=tokenizer,
            )
        else:
            pair_diagnostics = token_alignment_diagnostics(
                base_df,
                tokenizer=tokenizer,
            )
            filtered_pairs = filter_token_aligned_pairs(base_df, tokenizer)
            ensure_dir(pair_path.parent)
            save_csv(filtered_pairs, pair_path, index=False)

        if accepted_df.empty:
            filtered_accepted_pairs = accepted_df.copy()
        elif jsonl_pair_path.is_file() and not refresh:
            filtered_accepted_pairs = load_accepted_minimal_pair_dataframe(
                project_root,
                jsonl_pair_path,
            )
        else:
            filtered_accepted_pairs = filter_token_aligned_pairs(
                accepted_df,
                tokenizer,
            )
            write_accepted_minimal_pair_jsonl(filtered_accepted_pairs, jsonl_pair_path)

        animate_words, inanimate_words, target_summary, saved_target_path = (
            load_or_filter_targets_for_models(
                project_root,
                [model_name],
                target_tokenizer=tokenizer,
                refresh=refresh,
                target_source=target_source,
            )
        )
        filtered_pair_frames.append(filtered_pairs)
        filtered_accepted_frames.append(filtered_accepted_pairs)
        model_artifacts[model_name] = {
            "pairs_path": str(pair_path),
            "accepted_pairs_path": str(jsonl_pair_path),
            "targets_path": str(saved_target_path or target_path),
            "pair_count": int(len(filtered_pairs)),
            "accepted_pair_count": int(len(filtered_accepted_pairs)),
            "pair_retention_rate": (
                float(len(filtered_pairs) / len(base_df)) if len(base_df) else 0.0
            ),
            "accepted_pair_retention_rate": (
                float(len(filtered_accepted_pairs) / len(accepted_df))
                if len(accepted_df)
                else 0.0
            ),
            "target_counts": {
                "animate": int(len(animate_words)),
                "inanimate": int(len(inanimate_words)),
            },
            "pair_diagnostics": pair_diagnostics,
            "target_summary": target_summary,
        }

    intersection_prompt_pairs = intersect_prompt_pair_frames(filtered_pair_frames)
    intersection_pairs = filter_df_to_prompt_pairs(base_df, intersection_prompt_pairs)
    intersection_pair_path = tokenization_filter_intersection_pairs_path(
        project_root,
        resolved_model_names,
    )
    ensure_dir(intersection_pair_path.parent)
    if refresh or not intersection_pair_path.is_file():
        save_csv(intersection_pairs, intersection_pair_path, index=False)

    accepted_intersection_prompt_pairs = intersect_prompt_pair_frames(filtered_accepted_frames)
    accepted_intersection_pairs = filter_df_to_prompt_pairs(
        accepted_df,
        accepted_intersection_prompt_pairs,
    ) if not accepted_df.empty else accepted_df.copy()
    accepted_intersection_path = tokenization_filter_intersection_jsonl_pairs_path(
        project_root,
        resolved_model_names,
    )
    if refresh or not accepted_intersection_path.is_file():
        write_accepted_minimal_pair_jsonl(accepted_intersection_pairs, accepted_intersection_path)

    intersection_targets = filter_targets_for_models(
        project_root,
        resolved_model_names,
        target_source=target_source,
    )
    intersection_target_path = save_filtered_targets(
        project_root,
        resolved_model_names,
        intersection_targets,
        target_source=target_source,
    )
    summary = {
        "filter_version": TOKENIZATION_FILTER_VERSION,
        "target_source": str(target_source or DEFAULT_TARGET_SOURCE),
        "target_source_path": str(resolve_target_source_path(project_root, target_source)),
        "model_names": resolved_model_names,
        "base_pair_count": int(len(base_df)),
        "accepted_pair_count": int(len(accepted_df)),
        "models": model_artifacts,
        "intersection": {
            "pairs_path": str(intersection_pair_path),
            "accepted_pairs_path": str(accepted_intersection_path),
            "targets_path": str(intersection_target_path),
            "pair_count": int(len(intersection_pairs)),
            "accepted_pair_count": int(len(accepted_intersection_pairs)),
            "pair_retention_rate": (
                float(len(intersection_pairs) / len(base_df)) if len(base_df) else 0.0
            ),
            "accepted_pair_retention_rate": (
                float(len(accepted_intersection_pairs) / len(accepted_df))
                if len(accepted_df)
                else 0.0
            ),
            "target_counts": {
                "animate": int(len(intersection_targets["animate"])),
                "inanimate": int(len(intersection_targets["inanimate"])),
            },
            "target_summary": intersection_targets["summary"],
        },
    }
    summary_path = tokenization_filter_summary_path(project_root)
    save_json(summary_path, summary)
    return {
        "summary": summary,
        "paths": {
            "summary": str(summary_path),
            "intersection_pairs": str(intersection_pair_path),
            "accepted_intersection_pairs": str(accepted_intersection_path),
            "intersection_targets": str(intersection_target_path),
        },
    }


def metric_investigation_dir(project_root: Path, day: str | None = None) -> Path:
    return ensure_dir(results_root(project_root) / "metric_investigation" / (date_tag() if day is None else day))


def metric_investigation_model_dir(
    project_root: Path,
    model_name: str,
    target_source: str | Path | None = None,
) -> Path:
    base_dir = metric_filtered_model_dataset_dir(project_root, model_name)
    label = target_source_label(project_root, target_source).rstrip("_")
    return base_dir if not label else ensure_dir(base_dir / label)


def metric_investigation_scored_path(
    project_root: Path,
    model_name: str,
    target_source: str | Path | None = None,
) -> Path:
    if target_source_slug(project_root, target_source) == DEFAULT_TARGET_SOURCE:
        return metric_scored_model_dataset_path(project_root, model_name)
    model_slug = safe_model_name(canonical_model_name(model_name))
    return metric_investigation_model_dir(
        project_root,
        model_name,
        target_source,
    ) / f"mp_scored_metrics_{model_slug}.csv"


def metric_investigation_filtered_path(
    project_root: Path,
    model_name: str,
    metric_name: str,
    target_source: str | Path | None = None,
) -> Path:
    if target_source_slug(project_root, target_source) == DEFAULT_TARGET_SOURCE:
        return metric_filtered_model_dataset_path(project_root, model_name, metric_name)
    return metric_investigation_model_dir(
        project_root,
        model_name,
        target_source,
    ) / f"mp_filtered_{metric_name}.csv"


def metric_investigation_discovery_candidates_path(
    project_root: Path,
    model_name: str,
    metric_name: str,
    target_source: str | Path | None = None,
) -> Path:
    return metric_investigation_model_dir(
        project_root,
        model_name,
        target_source,
    ) / f"mp_discovery_candidates_{metric_name}.csv"


def save_metric_investigation_outputs(
    project_root: Path,
    model_name: str,
    scored_df: pd.DataFrame,
    margin_threshold: float,
    target_source: str | Path | None = None,
) -> dict[str, str]:
    resolved_model_name = canonical_model_name(model_name)
    scored_path = metric_investigation_scored_path(
        project_root,
        resolved_model_name,
        target_source=target_source,
    )
    ensure_dir(scored_path.parent)
    save_csv(scored_df, scored_path, index=False)

    paths = {"scored_metrics": str(scored_path)}
    filtered_by_metric: dict[str, pd.DataFrame] = {}
    for metric_name in METRIC_INVESTIGATION_METRICS:
        invalid_mask = (
            (scored_df[f"clean_{metric_name}"] < 0)
            | (scored_df[f"corrupt_{metric_name}"] > 0)
        )
        margin = scored_df[f"clean_{metric_name}"] - scored_df[f"corrupt_{metric_name}"]
        filtered_df = scored_df.loc[~invalid_mask].copy()
        filtered_by_metric[metric_name] = filtered_df
        filtered_path = metric_investigation_filtered_path(
            project_root,
            resolved_model_name,
            metric_name,
            target_source=target_source,
        )
        save_csv(filtered_df, filtered_path, index=False)
        paths[f"filtered_{metric_name}"] = str(filtered_path)

        discovery_candidates = filtered_df.loc[margin.loc[filtered_df.index] > margin_threshold].copy()
        discovery_candidates_path = metric_investigation_discovery_candidates_path(
            project_root,
            resolved_model_name,
            metric_name,
            target_source=target_source,
        )
        save_csv(discovery_candidates, discovery_candidates_path, index=False)
        paths[f"discovery_candidates_{metric_name}"] = str(discovery_candidates_path)

    success_df = filtered_by_metric[CHOSEN_DATASET_METRIC].assign(
        filter_model_name=resolved_model_name,
        filter_clean_metric=lambda data: data["clean_metric"],
        filter_corrupt_metric=lambda data: data["corrupt_metric"],
        filter_metric_name=CHOSEN_DATASET_METRIC,
        filter_seq_len=lambda data: data["seq_len"],
    )
    success_path = model_success_dataset_path(
        project_root,
        resolved_model_name,
        target_source=target_source,
    )
    ensure_dir(success_path.parent)
    save_csv(success_df, success_path, index=False)
    paths["model_success_cache"] = str(success_path)
    return paths


def run_metric_investigation_scoring(
    project_root: Path,
    model_names: Sequence[str] = DEFAULT_TOKENIZATION_FILTER_MODELS,
    *,
    batch_size: int = 32,
    top_k: int = 50,
    margin_threshold: float = 0.5,
    refresh_tokenization_filters: bool = False,
    output_day: str | None = None,
    target_source: str | Path | None = None,
) -> dict[str, Any]:
    resolved_model_names = unique_model_names(model_names)
    tokenization_filter_model_names = unique_model_names(DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS)
    day = output_day or date_tag()
    tokenization_artifact = prepare_tokenization_filter_artifacts(
        project_root=project_root,
        model_names=tokenization_filter_model_names,
        refresh=refresh_tokenization_filters,
        target_source=target_source,
    )
    pair_path = Path(tokenization_artifact["paths"]["intersection_pairs"])
    pair_df = pd.read_csv(pair_path)
    model_results: dict[str, Any] = {}
    success_pair_frames: list[pd.DataFrame] = []

    for model_name in resolved_model_names:
        print(
            f"Scoring {model_name} on {len(pair_df):,} common tokenizer-filtered pairs."
        )
        context: dict[str, Any] | None = None
        try:
            context = load_model_context(
                project_root,
                model_name,
                target_filter_model_names=tokenization_filter_model_names,
                target_source=target_source,
            )
            scored_df = compute_metric_investigation_scores(
                pair_df,
                model=context["model"],
                tokenizer=context["tokenizer"],
                animate_ids_tensor=context["animate_ids_tensor"],
                inanimate_ids_tensor=context["inanimate_ids_tensor"],
                batch_size=batch_size,
                top_k=top_k,
            )
            output_paths = save_metric_investigation_outputs(
                project_root,
                model_name,
                scored_df,
                margin_threshold=margin_threshold,
                target_source=target_source,
            )
            success_df = pd.read_csv(output_paths[f"filtered_{CHOSEN_DATASET_METRIC}"])
            success_pair_frames.append(prompt_pair_columns(success_df))
            model_results[model_name] = {
                "status": "scored",
                "pair_path": str(pair_path),
                "pair_count": int(len(pair_df)),
                "scored_count": int(len(scored_df)),
                "success_count": int(len(success_df)),
                "success_metric": CHOSEN_DATASET_METRIC,
                "discovery_margin_threshold": margin_threshold,
                "target_filter_path": context["target_filter_path"],
                "target_source": context["target_source"],
                "target_source_path": context["target_source_path"],
                "target_counts": {
                    "animate": int(len(context["animate_words"])),
                    "inanimate": int(len(context["inanimate_words"])),
                },
                "accuracy": task_accuracy_summary(scored_df),
                "paths": output_paths,
            }
        except Exception as exc:
            model_results[model_name] = {
                "status": "failed",
                "pair_path": str(pair_path),
                "pair_count": int(len(pair_df)),
                "error": repr(exc),
            }
            print(f"Failed scoring {model_name}: {exc!r}")
        finally:
            if context is not None:
                model = context.get("model")
                del context
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    success_intersection = intersect_prompt_pair_frames(success_pair_frames)
    output_dir = metric_investigation_dir(project_root, day)
    summary_path = output_dir / f"metric_investigation_scoring_{timestamp_tag()}.json"
    artifact = {
        "experiment": "metric_investigation_score",
        "config": {
            "model_names": resolved_model_names,
            "tokenization_filter_model_names": tokenization_filter_model_names,
            "batch_size": batch_size,
            "top_k": top_k,
            "discovery_margin_threshold": margin_threshold,
            "refresh_tokenization_filters": refresh_tokenization_filters,
            "output_day": day,
            "target_source": str(target_source or DEFAULT_TARGET_SOURCE),
            "target_source_path": str(resolve_target_source_path(project_root, target_source)),
        },
        "paths": {
            "project_root": str(project_root),
            "output_dir": str(output_dir),
            "summary": str(summary_path),
            "tokenization_filter_summary": tokenization_artifact["paths"]["summary"],
        },
        "model_results": model_results,
        "intersection_counts": {
            "tokenizer_metric_pair_count": int(
                tokenization_artifact["summary"]["intersection"]["pair_count"]
            ),
            "tokenizer_accepted_pair_count": int(
                tokenization_artifact["summary"]["intersection"]["accepted_pair_count"]
            ),
            "animate_target_count": int(
                tokenization_artifact["summary"]["intersection"]["target_counts"]["animate"]
            ),
            "inanimate_target_count": int(
                tokenization_artifact["summary"]["intersection"]["target_counts"]["inanimate"]
            ),
            "model_success_pair_count": int(len(success_intersection)),
        },
    }
    save_json(summary_path, artifact)
    return artifact


def intersect_target_scores_with_source_success(
    target_scored_df: pd.DataFrame,
    source_success_df: pd.DataFrame,
    dataset_filter_model_name: str,
) -> pd.DataFrame:
    base_columns = ["clean_prefix", "corrupt_prefix"]
    source_success_df = normalize_model_success_metadata(
        source_success_df,
        dataset_filter_model_name,
    )
    metadata_columns = [
        column
        for column in (
            "filter_model_name",
            "filter_clean_metric",
            "filter_corrupt_metric",
            "filter_seq_len",
        )
        if column in source_success_df.columns
    ]
    return target_scored_df.merge(
        source_success_df[base_columns + metadata_columns],
        on=base_columns,
        how="inner",
    )


def split_discovery_validation(
    df: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation_size = max(1, int(round(len(shuffled) * validation_fraction)))
    validation = shuffled.iloc[:validation_size].reset_index(drop=True).copy()
    discovery = shuffled.iloc[validation_size:].reset_index(drop=True).copy()
    if discovery.empty:
        raise ValueError("Discovery split is empty. Reduce validation_fraction or add more data.")
    return discovery, validation


def make_avg_logit_difference_recovery_metric(
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    eps: float = 1e-6,
):
    def metric_factory(clean_logits: torch.Tensor, corrupt_logits: torch.Tensor):
        clean_logit_diff = final_token_average_logit_difference(
            clean_logits,
            animate_ids_tensor,
            inanimate_ids_tensor,
        )
        corrupt_logit_diff = final_token_average_logit_difference(
            corrupt_logits,
            animate_ids_tensor,
            inanimate_ids_tensor,
        )
        denominator = clean_logit_diff - corrupt_logit_diff

        if not torch.all(denominator > eps):
            raise ValueError(
                "Each sample must have a sufficiently positive clean-corrupt logit-difference margin."
            )

        def metric(logits: torch.Tensor) -> torch.Tensor:
            patched_logit_diff = final_token_average_logit_difference(
                logits,
                animate_ids_tensor,
                inanimate_ids_tensor,
            )
            normalized_recovery = (patched_logit_diff - corrupt_logit_diff) / denominator
            if not torch.isfinite(normalized_recovery).all():
                raise ValueError("Non-finite values generated during normalized recovery calculation.")
            return normalized_recovery.mean()

        return metric, clean_logit_diff.mean().item(), corrupt_logit_diff.mean().item()

    return metric_factory


def make_eap_normalized_recovery_metric(
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
        if not torch.all(denominator > eps):
            raise ValueError(
                "Each sample must have a sufficiently positive clean-corrupt logit-difference margin."
            )

        recovery = (patched_logit_diff - corrupt_logit_diff) / denominator
        return recovery.mean()

    return metric


def make_eap_normalized_recovery_vector_metric(
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
        if not torch.all(denominator > eps):
            raise ValueError(
                "Each sample must have a sufficiently positive clean-corrupt logit-difference margin."
            )

        return (patched_logit_diff - corrupt_logit_diff) / denominator

    return metric


def make_eap_accuracy_metric(
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
):
    def metric(
        logits: torch.Tensor,
        clean_logits: torch.Tensor | None,
        input_lengths: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        del clean_logits, input_lengths, label
        patched_logit_diff = final_token_average_logit_difference(
            logits,
            animate_ids_tensor,
            inanimate_ids_tensor,
        )
        return (patched_logit_diff > 0).to(dtype=torch.float32)

    return metric


def _standardize_patch_result_shape(
    batch_patch_results: torch.Tensor,
    seq_len: int,
    n_heads: int,
) -> torch.Tensor:
    if batch_patch_results.ndim == 2:
        if batch_patch_results.shape[1] != seq_len:
            raise ValueError(
                f"Expected [layers, pos] with pos={seq_len}, got {tuple(batch_patch_results.shape)}."
            )
        return batch_patch_results

    if batch_patch_results.ndim == 3:
        if batch_patch_results.shape[1] == seq_len and batch_patch_results.shape[2] == n_heads:
            return batch_patch_results.permute(0, 2, 1)
        if batch_patch_results.shape[1] == n_heads and batch_patch_results.shape[2] == seq_len:
            return batch_patch_results
        raise ValueError(
            "3D patch result must have shape [layers, pos, heads] or [layers, heads, pos]."
        )

    raise ValueError(
        f"Patch result must be 2D or 3D, got {batch_patch_results.ndim}D."
    )


def batched_exact_patching(
    model: HookedTransformer,
    df: pd.DataFrame,
    tokenizer,
    patching_func,
    patching_metric_factory,
    names_filter,
    batch_size: int,
    requires_grad: bool = False,
    safety_checks: bool = False,
):
    total_samples = len(df)
    max_seq_len = int(df["seq_len"].max())

    total_clean_metric = 0.0
    total_corrupt_metric = 0.0
    total_patch_heatmap = None
    total_counts = None

    print(f"Starting batched patching over {total_samples} samples. Max sequence length: {max_seq_len}")

    context = torch.no_grad() if not requires_grad else contextlib.nullcontext()
    with context:
        func_name = getattr(
            patching_func,
            "__name__",
            getattr(getattr(patching_func, "func", None), "__name__", "patching_func"),
        )
        grouped = df.groupby("seq_len")

        with tqdm(total=total_samples, desc=f"Patching {func_name}", unit="seq") as progress:
            for length, group in grouped:
                if safety_checks:
                    sample_text = group.iloc[0]["clean_prefix"]
                    sample_tokens = get_input_ids_with_bos(
                        sample_text,
                        model,
                    )
                    final_token = tokenizer.decode(sample_tokens[-1])
                    print(f"[DEBUG] Group Length {length} | Example: '{sample_text}'")
                    print(
                        "[DEBUG]        ---> logits[:, -1, :] predicts the NEXT token after "
                        f"the final input token '{final_token}'\n"
                    )

                for start in range(0, len(group), batch_size):
                    batch_df = group.iloc[start : start + batch_size]
                    actual_batch_size = len(batch_df)

                    clean_batch = get_input_ids_with_bos(
                        batch_df["clean_prefix"].tolist(),
                        model,
                    ).to(model.cfg.device)
                    corrupt_batch = get_input_ids_with_bos(
                        batch_df["corrupt_prefix"].tolist(),
                        model,
                    ).to(model.cfg.device)

                    if clean_batch.shape != corrupt_batch.shape:
                        raise ValueError(
                            f"Shape mismatch: clean={tuple(clean_batch.shape)} corrupt={tuple(corrupt_batch.shape)}"
                        )

                    clean_logits, clean_cache = model.run_with_cache(
                        clean_batch,
                        names_filter=names_filter,
                    )
                    corrupt_logits = model(corrupt_batch)

                    batch_metric, clean_base, corrupt_base = patching_metric_factory(
                        clean_logits,
                        corrupt_logits,
                    )
                    total_clean_metric += clean_base * actual_batch_size
                    total_corrupt_metric += corrupt_base * actual_batch_size

                    batch_patch_results = patching_func(
                        model=model,
                        corrupted_tokens=corrupt_batch,
                        clean_cache=clean_cache,
                        patching_metric=batch_metric,
                    )
                    batch_patch_results = _standardize_patch_result_shape(
                        batch_patch_results=batch_patch_results,
                        seq_len=length,
                        n_heads=model.cfg.n_heads,
                    )

                    if total_patch_heatmap is None:
                        target_shape = list(batch_patch_results.shape)
                        target_shape[-1] = max_seq_len
                        total_patch_heatmap = torch.zeros(
                            target_shape,
                            device=batch_patch_results.device,
                            dtype=batch_patch_results.dtype,
                        )
                        total_counts = torch.zeros_like(total_patch_heatmap)

                    total_patch_heatmap[..., max_seq_len - length :] += (
                        batch_patch_results * actual_batch_size
                    )
                    total_counts[..., max_seq_len - length :] += actual_batch_size

                    del clean_logits, corrupt_logits, clean_cache, batch_patch_results
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    progress.update(actual_batch_size)

    final_clean_metric = total_clean_metric / total_samples
    final_corrupt_metric = total_corrupt_metric / total_samples

    counts_safe = torch.where(total_counts == 0, torch.ones_like(total_counts), total_counts)
    normalized_heatmap = total_patch_heatmap / counts_safe
    return normalized_heatmap, final_clean_metric, final_corrupt_metric


def run_and_save_experiment(
    project_root: Path,
    model: HookedTransformer,
    experiment_name: str,
    df: pd.DataFrame,
    tokenizer,
    patching_func,
    metric_factory,
    filter_str: str,
    batch_size: int,
    safety_checks: bool = False,
    output_dir: Path | None = None,
) -> Path:
    normalized_map, clean_metric_mean, corrupt_metric_mean = batched_exact_patching(
        model=model,
        df=df,
        tokenizer=tokenizer,
        patching_func=patching_func,
        patching_metric_factory=metric_factory,
        names_filter=lambda name: name.endswith(filter_str),
        batch_size=batch_size,
        safety_checks=safety_checks,
    )

    max_idx = df["seq_len"].idxmax()
    sample_tokens = get_input_ids_with_bos(
        df.loc[max_idx, "clean_prefix"],
        model,
    )

    experiment_data = {
        "experiment": experiment_name,
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d"),
        "normalized_heatmap": normalized_map.cpu(),
        "clean_metric_mean": clean_metric_mean,
        "corrupt_metric_mean": corrupt_metric_mean,
        "labels": [model.to_string(token) for token in sample_tokens],
    }

    save_dir = (
        manual_circuit_checkpoints_dir(
            project_root,
            getattr(model.cfg, "model_name", "unknown"),
            experiment_data["timestamp"],
        )
        if output_dir is None
        else ensure_dir(output_dir)
    )
    save_path = save_dir / f"{experiment_name}_{experiment_data['timestamp']}.pt"
    save_torch(experiment_data, save_path)
    return save_path


def get_saved_normalized_heatmap(data: dict[str, Any]) -> torch.Tensor:
    if "normalized_heatmap" in data:
        return data["normalized_heatmap"]
    return (data["raw_heatmap"] - data["corrupt_baseline"]) / (
        data["clean_baseline"] - data["corrupt_baseline"]
    )


def get_saved_heatmap_x_labels(data: dict[str, Any]) -> tuple[list[str] | None, str]:
    raw_labels = data.get("labels")
    if raw_labels is None:
        return None, "Token Position"

    stripped_labels = [label.strip() for label in raw_labels]
    leading_bos_tokens = {"<|endoftext|>", "<bos>", "<s>"}
    template_offset = 1 if stripped_labels and stripped_labels[0].lower() in leading_bos_tokens else 0
    template_slice = stripped_labels[template_offset:]
    if (
        len(template_slice) == 6
        and template_slice[0].lower() == "the"
        and template_slice[2].lower() == "was"
        and template_slice[4].lower() == "by"
        and template_slice[5].lower() == "the"
    ):
        template_labels = template_slice.copy()
        template_labels[1] = "[patient]"
        template_labels[3] = "[verb]"
        if template_offset == 1:
            return [stripped_labels[0], *template_labels], "Template Position"
        return template_labels, "Template Position"

    return stripped_labels, "Token Position"


def get_saved_token_index(
    data: dict[str, Any],
    token_label: str,
    occurrence: int = 0,
) -> int:
    plot_labels, _ = get_saved_heatmap_x_labels(data)
    matching_indices = [
        idx for idx, label in enumerate(plot_labels)
        if label.strip().lower() == token_label.strip().lower()
    ]
    if not matching_indices:
        raise ValueError(f"Could not find token '{token_label}' in labels: {plot_labels}")

    try:
        return matching_indices[occurrence]
    except IndexError as exc:
        raise ValueError(
            f"Token '{token_label}' does not have occurrence {occurrence} in labels: {plot_labels}"
        ) from exc


def heatmap_to_score_rows(data: dict[str, Any]) -> pd.DataFrame:
    heatmap = get_saved_normalized_heatmap(data)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {tuple(heatmap.shape)}")

    labels, _ = get_saved_heatmap_x_labels(data)
    if labels is None:
        labels = [str(index) for index in range(heatmap.shape[1])]

    rows: list[dict[str, Any]] = []
    for token_position in range(heatmap.shape[1]):
        token = labels[token_position]
        pos_from_end = token_position - heatmap.shape[1]
        for layer in range(heatmap.shape[0]):
            rows.append(
                {
                    "layer": layer,
                    "token_position": token_position,
                    "token_position_from_end": pos_from_end,
                    "token": token,
                    "score": float(heatmap[layer, token_position].item()),
                }
            )
    return pd.DataFrame(rows)


def head_heatmap_to_score_rows(data: dict[str, Any]) -> pd.DataFrame:
    heatmap = get_saved_normalized_heatmap(data)
    if heatmap.ndim != 3:
        raise ValueError(f"Expected a 3D head heatmap, got shape {tuple(heatmap.shape)}")

    labels, _ = get_saved_heatmap_x_labels(data)
    if labels is None:
        raise ValueError("Saved head heatmap is missing token labels.")

    if heatmap.shape[2] == len(labels):
        normalized = heatmap
    elif heatmap.shape[1] == len(labels):
        normalized = heatmap.permute(0, 2, 1)
    else:
        raise ValueError(
            f"Could not infer token axis for head heatmap with shape {tuple(heatmap.shape)} "
            f"and {len(labels)} labels."
        )

    rows: list[dict[str, Any]] = []
    for token_position in range(normalized.shape[2]):
        token = labels[token_position]
        pos_from_end = token_position - normalized.shape[2]
        for layer in range(normalized.shape[0]):
            for head in range(normalized.shape[1]):
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "token_position": token_position,
                        "token_position_from_end": pos_from_end,
                        "token": token,
                        "score": float(normalized[layer, head, token_position].item()),
                    }
                )
    return pd.DataFrame(rows)


def draw_heatmap(project_root: Path, file_path: Path) -> None:
    data = torch.load(file_path)
    normalized_map = get_saved_normalized_heatmap(data)
    x_labels, x_axis_title = get_saved_heatmap_x_labels(data)
    fig = px.imshow(
        normalized_map.numpy(),
        labels={"x": x_axis_title, "y": "Layer"},
        x=x_labels,
        title=f"Results: {data['experiment']}",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    )
    fig.update_yaxes(autorange="reversed")
    fig.write_image(
        str(
            resolve_image_output_dir(project_root, file_path)
            / f"{data['experiment']}_head_heatmap.png"
        )
    )
    fig.show()


def make_multi_site_patching_func(
    sites: Sequence[PatchSite],
    component: str,
):
    grouped_sites: dict[int, list[PatchSite]] = defaultdict(list)
    for site in sites:
        grouped_sites[site.layer].append(site)

    def custom_patching_func(model, corrupted_tokens, clean_cache, patching_metric):
        seq_len = corrupted_tokens.shape[1]
        n_layers = model.cfg.n_layers
        results = torch.zeros((n_layers, seq_len), device=model.cfg.device)

        for layer, layer_sites in grouped_sites.items():
            hook_name = f"blocks.{layer}.hook_{component}"
            for site in layer_sites:
                pos = seq_len + site.token_position_from_end
                if pos < 0 or pos >= seq_len:
                    continue

                def single_pos_hook_fn(corrupt_act, hook, target_pos=pos):
                    clean_act = clean_cache[hook.name]
                    corrupt_act[:, target_pos, :] = clean_act[:, target_pos, :]
                    return corrupt_act

                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, single_pos_hook_fn)],
                    return_type="logits",
                )
                results[layer, pos] = patching_metric(patched_logits).item()

        return results

    return custom_patching_func


def positive_percentile_threshold(scores: pd.Series, percentile: int) -> float | None:
    positive = scores[scores > 0]
    if positive.empty:
        return None
    return float(np.percentile(positive.to_numpy(dtype=np.float64), percentile))


def select_positive_rows_at_percentile(
    df: pd.DataFrame,
    percentile: int,
) -> pd.DataFrame:
    threshold = positive_percentile_threshold(df["score"], percentile)
    if threshold is None:
        return df.iloc[0:0].copy()
    selected = df[(df["score"] > 0) & (df["score"] >= threshold)].copy()
    return selected.sort_values("score", ascending=False).reset_index(drop=True)


DISCOVERED_COMPONENT_SPECS = (
    {
        "name": "verb",
        "token_label": "[verb]",
        "occurrence": 0,
        "mlp_filename": "MLP_Outputs_Targeted_Verb_{day}.pt",
        "attn_filename": "Attention_Outputs_Targeted_Verb_{day}.pt",
    },
    {
        "name": "by",
        "token_label": "by",
        "occurrence": 0,
        "mlp_filename": "MLP_Outputs_Targeted_by_{day}.pt",
        "attn_filename": "Attention_Outputs_Targeted_by_{day}.pt",
    },
    {
        "name": "the",
        "token_label": "the",
        "occurrence": -1,
        "mlp_filename": "MLP_Outputs_Targeted_the_{day}.pt",
        "attn_filename": "Attention_Outputs_Targeted_the_{day}.pt",
    },
)


def resolve_day_result_path(results_dir: Path, day: str, filename_template: str) -> Path:
    filename = filename_template.format(day=day)
    experiment_name = filename.removesuffix(".pt").removesuffix(f"_{day}")
    candidates = [
        results_dir / day / filename,
        results_dir / experiment_name / "gpt2" / day / filename,
    ]
    candidates.extend(
        sorted(results_dir.glob(f"manual_circuit_discovery/*/{day}/checkpoints/{filename}"))
    )
    candidates.extend(sorted(results_dir.glob(f"*/gpt2/{day}/{filename}")))
    candidates.extend(sorted(results_dir.glob(f"*/*/{day}/{filename}")))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        raise FileNotFoundError(f"Could not find checkpoint: {results_dir / day / filename}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def _token_position_from_end(
    data: dict[str, Any],
    token_label: str,
    occurrence: int,
) -> int:
    token_idx = get_saved_token_index(data, token_label, occurrence=occurrence)
    return int(token_idx - get_saved_normalized_heatmap(data).shape[1])


def _positive_token_layer_rows(
    data: dict[str, Any],
    token_label: str,
    occurrence: int,
    min_score: float = 0.0,
) -> pd.DataFrame:
    rows = heatmap_to_score_rows(data)
    token_pos_from_end = _token_position_from_end(data, token_label, occurrence)
    selected = rows[
        (rows["token_position_from_end"] == token_pos_from_end)
        & (rows["score"] > min_score)
    ].copy()
    return selected.sort_values("score", ascending=False).reset_index(drop=True)


def build_head_layer_targets(
    results_dir: Path,
    day: str,
    min_score: float = 0.0,
) -> tuple[dict[int, list[int]], pd.DataFrame]:
    head_layer_targets: dict[int, list[int]] = {}
    summary_rows: list[dict[str, Any]] = []

    for spec in DISCOVERED_COMPONENT_SPECS:
        file_path = resolve_day_result_path(results_dir, day, spec["attn_filename"])
        data = torch.load(file_path)
        pos_from_end = _token_position_from_end(data, spec["token_label"], spec["occurrence"])
        positive_rows = _positive_token_layer_rows(
            data=data,
            token_label=spec["token_label"],
            occurrence=spec["occurrence"],
            min_score=min_score,
        )
        selected_layers = sorted({int(layer) for layer in positive_rows["layer"].tolist()})
        head_layer_targets[pos_from_end] = selected_layers

        for row in positive_rows.itertuples(index=False):
            summary_rows.append(
                {
                    "source": spec["name"],
                    "component_type": "attn_layer",
                    "token": spec["token_label"],
                    "token_position_from_end": int(row.token_position_from_end),
                    "layer": int(row.layer),
                    "score": float(row.score),
                    "checkpoint": str(file_path),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["token_position_from_end", "score"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return head_layer_targets, summary_df


def select_discovered_component_nodes(
    results_dir: Path,
    day: str,
    attention_head_checkpoint: Path,
    min_score: float = 0.0,
) -> tuple[set[str], pd.DataFrame]:
    component_frames: list[pd.DataFrame] = []
    source_by_pos_from_end: dict[int, str] = {}

    for spec in DISCOVERED_COMPONENT_SPECS:
        mlp_path = resolve_day_result_path(results_dir, day, spec["mlp_filename"])
        mlp_data = torch.load(mlp_path)
        pos_from_end = _token_position_from_end(mlp_data, spec["token_label"], spec["occurrence"])
        source_by_pos_from_end[pos_from_end] = spec["name"]

        mlp_rows = _positive_token_layer_rows(
            data=mlp_data,
            token_label=spec["token_label"],
            occurrence=spec["occurrence"],
            min_score=min_score,
        )
        if mlp_rows.empty:
            continue

        mlp_rows = mlp_rows.assign(
            source=spec["name"],
            component_type="mlp",
            node=mlp_rows["layer"].map(lambda layer: f"m{int(layer)}"),
            checkpoint=str(mlp_path),
        )
        component_frames.append(mlp_rows)

    head_layer_targets, _ = build_head_layer_targets(results_dir, day, min_score=min_score)
    head_rows = head_heatmap_to_score_rows(torch.load(attention_head_checkpoint))
    for pos_from_end, allowed_layers in head_layer_targets.items():
        if not allowed_layers:
            continue

        selected_head_rows = head_rows[
            (head_rows["token_position_from_end"] == pos_from_end)
            & (head_rows["layer"].isin(allowed_layers))
            & (head_rows["score"] > min_score)
        ].copy()
        if selected_head_rows.empty:
            continue

        selected_head_rows = selected_head_rows.assign(
            source=source_by_pos_from_end[pos_from_end],
            component_type="attn_head",
            node=selected_head_rows.apply(lambda row: f"a{int(row['layer'])}.h{int(row['head'])}", axis=1),
            checkpoint=str(attention_head_checkpoint),
        )
        component_frames.append(selected_head_rows)

    if not component_frames:
        return set(), pd.DataFrame()

    component_frame = pd.concat(component_frames, ignore_index=True)
    component_frame = component_frame.sort_values(
        ["component_type", "token_position_from_end", "score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    retained_nodes = set(component_frame["node"].tolist())
    return retained_nodes, component_frame


def patch_rows_to_sites(df: pd.DataFrame) -> list[PatchSite]:
    return [
        PatchSite(
            layer=int(row.layer),
            token_position_from_end=int(row.token_position_from_end),
            token=str(row.token),
            score=float(row.score),
        )
        for row in df.itertuples(index=False)
    ]


def build_graph(model: HookedTransformer) -> Graph:
    from eap.graph import Graph

    return Graph.from_model(model)


def clone_graph(graph: Graph) -> Graph:
    from eap.graph import Graph

    cloned = Graph.from_model(dict(graph.cfg))
    cloned.real_edge_mask[:] = graph.real_edge_mask.clone()
    cloned.scores[:] = graph.scores.clone()
    return cloned


def collapse_edge_name(edge_name: str) -> str:
    if "<" not in edge_name:
        return edge_name
    return edge_name.split("<", maxsplit=1)[0]


def collapsed_edge_groups(graph: Graph) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for edge_name, edge in graph.edges.items():
        if not bool(graph.real_edge_mask[edge.matrix_index].item()):
            continue

        collapsed_name = collapse_edge_name(edge_name)
        score = float(edge.score.item())
        group = groups.setdefault(
            collapsed_name,
            {
                "collapsed_edge": collapsed_name,
                "parent": edge.parent.name,
                "child": edge.child.name,
                "signed_sum": 0.0,
                "abs_score": 0.0,
                "underlying_edges": [],
            },
        )
        group["signed_sum"] += score
        group["abs_score"] += abs(score)
        group["underlying_edges"].append(edge_name)

    ranked = sorted(groups.values(), key=lambda item: item["abs_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["underlying_edge_count"] = len(item["underlying_edges"])
    return ranked


def induced_node_ranking(collapsed_edges: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    node_scores: dict[str, float] = defaultdict(float)
    for edge in collapsed_edges:
        node_scores[edge["parent"]] += float(edge["abs_score"])
        node_scores[edge["child"]] += float(edge["abs_score"])

    ranked = [
        {"node": node, "induced_score": score}
        for node, score in node_scores.items()
        if node not in {"input", "logits"}
    ]
    ranked.sort(key=lambda item: item["induced_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def collapsed_edge_count_from_mask(graph: Graph) -> int:
    seen = set()
    for edge_name, edge in graph.edges.items():
        if bool(graph.real_edge_mask[edge.matrix_index].item()):
            seen.add(collapse_edge_name(edge_name))
    return len(seen)


def build_reduced_graph(
    full_graph: Graph,
    retained_node_names: set[str],
) -> tuple[Graph, list[str]]:
    reduced = Graph.from_model(dict(full_graph.cfg))
    reduced.real_edge_mask[:] = False
    reduced.scores[:] = 0

    allowed_nodes = set(retained_node_names) | {"input", "logits"}
    allowed_edge_names: list[str] = []
    for edge_name, edge in reduced.edges.items():
        if not bool(full_graph.real_edge_mask[edge.matrix_index].item()):
            continue
        if edge.parent.name in allowed_nodes and edge.child.name in allowed_nodes:
            reduced.real_edge_mask[edge.matrix_index] = True
            allowed_edge_names.append(edge_name)
    return reduced, allowed_edge_names


def make_dataloader(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = CircuitPairDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def attribute_graph(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[..., torch.Tensor],
    ig_steps: int,
) -> Graph:
    from eap.attribute import attribute

    if not torch.cuda.is_available():
        raise EnvironmentError(
            "The installed EAP package requires a CUDA-enabled PyTorch build for attribution. "
            "Run this notebook in a CUDA environment before launching EAP-IG."
        )
    attribute(
        model,
        graph,
        dataloader,
        metric,
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
        quiet=False,
    )
    return graph


def run_component_eap(
    model: HookedTransformer,
    df: pd.DataFrame,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    retained_nodes: set[str],
    attribution_batch_size: int,
    ig_steps: int,
    output_dir: Path | None = None,
    output_prefix: str = "discovered_components_eap",
) -> dict[str, Any]:
    if not retained_nodes:
        raise ValueError("No retained nodes were provided for component EAP.")
    if "corrupt_metric" not in df.columns:
        raise ValueError("Dataframe must include a 'corrupt_metric' column for EAP attribution.")

    full_graph = build_graph(model)
    reduced_graph, allowed_edge_names = build_reduced_graph(full_graph, retained_nodes)
    if not allowed_edge_names:
        raise ValueError("The reduced graph has no allowed edges for the provided retained nodes.")

    dataloader = make_dataloader(df, batch_size=attribution_batch_size, shuffle=False)
    metric = make_eap_normalized_recovery_metric(animate_ids_tensor, inanimate_ids_tensor)
    scored_graph = attribute_graph(
        model=model,
        graph=reduced_graph,
        dataloader=dataloader,
        metric=metric,
        ig_steps=ig_steps,
    )

    ranked_edges = collapsed_edge_groups(scored_graph)
    ranked_nodes = induced_node_ranking(ranked_edges)
    edge_frame = ranking_frame(ranked_edges)
    node_frame = ranking_frame(ranked_nodes)

    if output_dir is not None:
        ensure_dir(output_dir)
        save_csv(edge_frame, output_dir / f"{output_prefix}_edges.csv", index=False)
        save_csv(node_frame, output_dir / f"{output_prefix}_nodes.csv", index=False)

    return {
        "graph": scored_graph,
        "ranked_edges": ranked_edges,
        "ranked_nodes": ranked_nodes,
        "edge_frame": edge_frame,
        "node_frame": node_frame,
        "allowed_edge_names": allowed_edge_names,
    }


def build_budget_circuit(
    scored_graph: Graph,
    ranked_collapsed_edges: Sequence[dict[str, Any]],
    budget: int,
) -> Graph:
    if budget > len(ranked_collapsed_edges):
        raise ValueError(f"Requested budget {budget} but only {len(ranked_collapsed_edges)} collapsed edges are available.")

    candidate = clone_graph(scored_graph)
    candidate.reset()
    for edge_group in ranked_collapsed_edges[:budget]:
        for edge_name in edge_group["underlying_edges"]:
            candidate.edges[edge_name].in_graph = True
    candidate.prune()
    return candidate


def evaluate_budget(
    model: HookedTransformer,
    scored_graph: Graph,
    ranked_collapsed_edges: Sequence[dict[str, Any]],
    validation_loader: DataLoader,
    metric: Callable[..., torch.Tensor],
    budget: int,
) -> BudgetEvaluation:
    from eap.evaluate import evaluate_graph

    candidate = build_budget_circuit(scored_graph, ranked_collapsed_edges, budget)
    results = evaluate_graph(
        model,
        candidate,
        validation_loader,
        metric,
        quiet=True,
        intervention="patching",
        skip_clean=False,
    )
    values = results.float().cpu()
    return BudgetEvaluation(
        budget=budget,
        faithfulness_mean=float(values.mean().item()),
        faithfulness_std=float(values.std(unbiased=False).item()) if len(values) > 1 else 0.0,
        example_count=int(len(values)),
        collapsed_edge_budget=budget,
        expanded_edge_count=int(candidate.count_included_edges()),
        induced_node_count=int(candidate.count_included_nodes() - 2),
    )


def ranking_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "underlying_edges" in frame.columns:
        frame = frame.copy()
        frame["underlying_edges"] = frame["underlying_edges"].apply(lambda values: "|".join(values))
    return frame


def parse_ranked_edge_frame(edge_frame: pd.DataFrame) -> list[dict[str, Any]]:
    ranked_edges: list[dict[str, Any]] = []
    for row in edge_frame.to_dict("records"):
        item = dict(row)
        underlying_edges = item.get("underlying_edges")
        if isinstance(underlying_edges, str):
            item["underlying_edges"] = [edge for edge in underlying_edges.split("|") if edge]
        elif underlying_edges is None:
            item["underlying_edges"] = []
        else:
            try:
                if pd.isna(underlying_edges):
                    item["underlying_edges"] = []
                    ranked_edges.append(item)
                    continue
            except TypeError:
                pass
            if isinstance(underlying_edges, Sequence) and not isinstance(underlying_edges, (str, bytes)):
                item["underlying_edges"] = list(underlying_edges)
            else:
                raise TypeError(f"Unsupported underlying_edges value: {underlying_edges!r}")
        ranked_edges.append(item)
    return ranked_edges


def load_saved_ranked_edges(
    edge_path: Path,
    node_path: Path | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame] | None:
    if not edge_path.exists():
        return None

    edge_frame = pd.read_csv(edge_path)
    if edge_frame.empty:
        return None

    node_frame = pd.read_csv(node_path) if node_path is not None and node_path.exists() else pd.DataFrame()
    ranked_edges = parse_ranked_edge_frame(edge_frame)
    if node_frame.empty:
        node_frame = ranking_frame(induced_node_ranking(ranked_edges))
    return ranked_edges, edge_frame, node_frame


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def resolve_shadow_source_artifacts(
    project_root: Path,
    model_name: str,
    dataset_set_name: str,
    main_experiment_path: str | Path | None = None,
) -> dict[str, Path | None]:
    if main_experiment_path is None:
        model_slug = safe_model_name(canonical_model_name(model_name))
        candidates = sorted(
            (
                project_root
                / "results"
                / "eap_ig"
                / model_slug
                / dataset_set_name
            ).glob("*/full_model"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                "No saved full-model EAP run found for "
                f"{model_slug}/{dataset_set_name}. Pass --main-experiment-path explicitly."
            )
        source_dir = candidates[-1]
    else:
        target = Path(main_experiment_path)
        if target.is_dir():
            source_dir = target
        elif target.is_file():
            source_dir = target.parent
        else:
            raise FileNotFoundError(f"Main experiment path does not exist: {target}")

    edge_path = _latest_matching_file(source_dir, "full_model_edges_*.csv")
    node_path = _latest_matching_file(source_dir, "full_model_nodes_*.csv")
    budget_path = _latest_matching_file(source_dir, "full_model_budget_sweep_*.csv")
    summary_path = _latest_matching_file(source_dir, "full_model_summary_*.json")

    if main_experiment_path is not None and Path(main_experiment_path).is_file():
        target = Path(main_experiment_path)
        if target.name.startswith("full_model_edges_") and target.suffix == ".csv":
            edge_path = target
        elif target.name.startswith("full_model_nodes_") and target.suffix == ".csv":
            node_path = target
        elif target.name.startswith("full_model_budget_sweep_") and target.suffix == ".csv":
            budget_path = target
        elif target.name.startswith("full_model_summary_") and target.suffix == ".json":
            summary_path = target

    missing = []
    if edge_path is None:
        missing.append("full_model_edges_*.csv")
    if budget_path is None:
        missing.append("full_model_budget_sweep_*.csv")
    if missing:
        raise FileNotFoundError(
            f"Missing required source artifact(s) in {source_dir}: {', '.join(missing)}"
        )

    return {
        "source_dir": source_dir,
        "edge_path": edge_path,
        "node_path": node_path,
        "budget_path": budget_path,
        "summary_path": summary_path,
    }


def first_budget_reaching_faithfulness(
    budget_frame: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    required = {"collapsed_edge_budget", "faithfulness_mean"}
    missing = sorted(required - set(budget_frame.columns))
    if missing:
        raise ValueError(f"Budget sweep is missing required columns: {missing}")

    ordered = budget_frame.copy()
    ordered["collapsed_edge_budget"] = ordered["collapsed_edge_budget"].astype(int)
    ordered["faithfulness_mean"] = ordered["faithfulness_mean"].astype(float)
    ordered = ordered.sort_values("collapsed_edge_budget", kind="stable")
    reached = ordered[ordered["faithfulness_mean"] >= float(threshold)]
    if reached.empty:
        max_value = float(ordered["faithfulness_mean"].max()) if not ordered.empty else 0.0
        raise ValueError(
            f"No budget reaches faithfulness {threshold:.3f}; max observed faithfulness is {max_value:.6f}."
        )
    row = reached.iloc[0].to_dict()
    row["collapsed_edge_budget"] = int(row["collapsed_edge_budget"])
    row["faithfulness_mean"] = float(row["faithfulness_mean"])
    return row


def select_top_edge_groups(
    ranked_edges: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("Edge count must be positive.")
    if count > len(ranked_edges):
        raise ValueError(f"Requested {count} edges but only {len(ranked_edges)} are available.")
    return list(ranked_edges[:count])


def underlying_edge_name_set(edge_groups: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(edge_name)
        for edge_group in edge_groups
        for edge_name in edge_group.get("underlying_edges", [])
    }


def build_edge_removed_graph(
    model: HookedTransformer,
    removed_edge_groups: Sequence[dict[str, Any]],
) -> tuple[Graph, list[str]]:
    graph = build_graph(model)
    removed_underlying = underlying_edge_name_set(removed_edge_groups)
    missing = sorted(edge_name for edge_name in removed_underlying if edge_name not in graph.edges)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Source ranking contains edges absent from the current graph: {preview}{suffix}")
    for edge_name in removed_underlying:
        edge = graph.edges[edge_name]
        graph.real_edge_mask[edge.matrix_index] = False
        edge.in_graph = False
    return graph, sorted(removed_underlying)


def edge_overlap_summary(
    rediscovered_edges: Sequence[dict[str, Any]],
    source_edges: Sequence[dict[str, Any]],
    removed_edges: Sequence[dict[str, Any]],
    top_k_values: Sequence[int] = (100, 500, 1000),
) -> dict[str, Any]:
    source_names = [str(edge["collapsed_edge"]) for edge in source_edges]
    removed_names = {str(edge["collapsed_edge"]) for edge in removed_edges}
    rediscovered_names = [str(edge["collapsed_edge"]) for edge in rediscovered_edges]
    summary: dict[str, Any] = {
        "removed_collapsed_edge_count": int(len(removed_names)),
        "rediscovered_ranked_edge_count": int(len(rediscovered_names)),
        "removed_edges_rediscovered_count": int(len(set(rediscovered_names) & removed_names)),
    }
    for top_k in top_k_values:
        k = int(top_k)
        source_top = set(source_names[:k])
        rediscovered_top = set(rediscovered_names[:k])
        denominator = len(source_top | rediscovered_top)
        summary[f"top_{k}_source_overlap_count"] = int(len(source_top & rediscovered_top))
        summary[f"top_{k}_source_jaccard"] = (
            float(len(source_top & rediscovered_top) / denominator) if denominator else 0.0
        )
        summary[f"top_{k}_removed_overlap_count"] = int(len(rediscovered_top & removed_names))
    return summary


def validate_shadow_source_provenance(
    *,
    source_summary_path: Path | None,
    prepared: dict[str, Any],
    config: EAPShadowRediscoveryConfig,
) -> dict[str, Any]:
    if source_summary_path is None:
        return {"status": "unverified_no_summary", "reason": "source summary JSON not found"}

    summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    dataset_summary = summary.get("dataset_summary", {})
    summary_config = summary.get("config", {})
    mismatches: list[str] = []

    if dataset_summary.get("dataset_set_name") != config.dataset_set_name:
        mismatches.append(
            f"dataset_set_name={dataset_summary.get('dataset_set_name')!r} != {config.dataset_set_name!r}"
        )
    if canonical_model_name(dataset_summary.get("target_model", config.model_name)) != canonical_model_name(prepared["model_name"]):
        mismatches.append(
            f"target_model={dataset_summary.get('target_model')!r} != {prepared['model_name']!r}"
        )
    if dataset_summary.get("target_filter_policy", config.target_filter_policy) != config.target_filter_policy:
        mismatches.append(
            f"target_filter_policy={dataset_summary.get('target_filter_policy')!r} != {config.target_filter_policy!r}"
        )
    if int(summary_config.get("seed", config.seed)) != int(config.seed):
        mismatches.append(f"seed={summary_config.get('seed')!r} != {config.seed}")
    if int(dataset_summary.get("discovery_count", config.discovery_sample_size)) != int(config.discovery_sample_size):
        mismatches.append(
            f"discovery_count={dataset_summary.get('discovery_count')!r} != {config.discovery_sample_size}"
        )
    if dataset_summary.get("discovery_sample_signature") != prepared["sample_signature"]:
        mismatches.append("discovery_sample_signature mismatch")
    expected_prompt_signature = dataset_summary.get("prompt_pair_signature")
    if expected_prompt_signature is not None and expected_prompt_signature != prompt_pair_signature(prepared["filtered_df"]):
        mismatches.append("prompt_pair_signature mismatch")

    if mismatches:
        raise ValueError("Source full-model EAP run does not match this rediscovery run: " + "; ".join(mismatches))

    return {
        "status": "verified",
        "summary_path": str(source_summary_path),
        "experiment": summary.get("experiment"),
    }


def save_rankings(
    output_dir: Path,
    prefix: str,
    collapsed_edges: Sequence[dict[str, Any]],
    nodes: Sequence[dict[str, Any]],
) -> None:
    save_csv(ranking_frame(collapsed_edges), output_dir / f"{prefix}_edges.csv", index=False)
    save_csv(ranking_frame(nodes), output_dir / f"{prefix}_nodes.csv", index=False)


def eap_node_metadata(node_name: str) -> dict[str, Any]:
    if node_name == "input":
        return {"kind": "input", "layer": -1, "head": None}
    if node_name == "logits":
        return {"kind": "logits", "layer": -1, "head": None}

    mlp_match = re.fullmatch(r"m(\d+)", node_name)
    if mlp_match:
        return {"kind": "mlp", "layer": int(mlp_match.group(1)), "head": None}

    attn_match = re.fullmatch(r"a(\d+)\.h(\d+)", node_name)
    if attn_match:
        return {
            "kind": "attn",
            "layer": int(attn_match.group(1)),
            "head": int(attn_match.group(2)),
        }

    return {"kind": "other", "layer": 0, "head": None}


def eap_node_sort_key(node_name: str) -> tuple[int, int, int, str]:
    meta = eap_node_metadata(node_name)
    type_order = {"input": 0, "attn": 1, "mlp": 2, "logits": 3, "other": 4}[meta["kind"]]
    head_order = -1 if meta["head"] is None else int(meta["head"])
    return (int(meta["layer"]), type_order, head_order, node_name)


def build_layered_circuit_figure(edge_frame: pd.DataFrame, top_k: int = 40) -> go.Figure:
    if edge_frame.empty:
        raise ValueError("edge_frame is empty. Run EAP attribution before visualizing the circuit.")

    top_edges = edge_frame.nlargest(top_k, "abs_score").copy()
    nodes = sorted(set(top_edges["parent"]).union(top_edges["child"]), key=eap_node_sort_key)
    metadata = {node: eap_node_metadata(node) for node in nodes}

    layer_values = [
        int(meta["layer"])
        for meta in metadata.values()
        if meta["kind"] in {"attn", "mlp"}
    ]
    max_layer = max(layer_values) if layer_values else 0

    incident_strength: dict[str, float] = defaultdict(float)
    for row in top_edges.itertuples(index=False):
        incident_strength[str(row.parent)] += float(row.abs_score)
        incident_strength[str(row.child)] += float(row.abs_score)

    buckets: dict[Any, list[str]] = defaultdict(list)
    for node in nodes:
        node_meta = metadata[node]
        bucket_key = (
            node_meta["kind"]
            if node_meta["kind"] in {"input", "logits"}
            else int(node_meta["layer"])
        )
        buckets[bucket_key].append(node)

    positions: dict[str, tuple[float, float]] = {}
    for bucket_nodes in buckets.values():
        bucket_nodes.sort(key=eap_node_sort_key)
        center = (len(bucket_nodes) - 1) / 2
        for idx, node in enumerate(bucket_nodes):
            node_meta = metadata[node]
            if node_meta["kind"] == "input":
                x = 0.0
            elif node_meta["kind"] == "logits":
                x = float(max_layer + 2)
            elif node_meta["kind"] == "attn":
                x = float(int(node_meta["layer"]) + 1) - 0.18
            elif node_meta["kind"] == "mlp":
                x = float(int(node_meta["layer"]) + 1) + 0.18
            else:
                x = float(int(node_meta["layer"]) + 1)
            positions[node] = (x, float(center - idx))

    color_map = {
        "input": "#7f7f7f",
        "attn": "#1f77b4",
        "mlp": "#ff7f0e",
        "logits": "#2f2f2f",
        "other": "#9467bd",
    }
    max_incident = max(incident_strength.values()) if incident_strength else 1.0
    max_edge_score = float(top_edges["abs_score"].max()) if len(top_edges) else 1.0

    fig = go.Figure()
    for row in top_edges.sort_values("abs_score", ascending=False).itertuples(index=False):
        x0, y0 = positions[str(row.parent)]
        x1, y1 = positions[str(row.child)]
        edge_color = (
            "rgba(31,119,180,0.70)"
            if float(row.signed_sum) >= 0
            else "rgba(214,39,40,0.70)"
        )
        edge_width = 1.5 + 8.0 * (
            float(row.abs_score) / max_edge_score if max_edge_score else 0.0
        )
        hover_text = (
            f"Rank {int(row.rank)}<br>"
            f"{row.parent} -> {row.child}<br>"
            f"Signed score: {float(row.signed_sum):.4f}<br>"
            f"|score|: {float(row.abs_score):.4f}<br>"
            f"Underlying edges: {int(row.underlying_edge_count)}"
        )
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
            )
        )

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_color: list[str] = []
    node_size: list[float] = []
    node_hover: list[str] = []
    for node in nodes:
        node_meta = metadata[node]
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(color_map.get(str(node_meta["kind"]), color_map["other"]))
        node_size.append(
            14 + 18 * (incident_strength[node] / max_incident if max_incident else 0.0)
        )
        node_hover.append(
            "<br>".join(
                [
                    f"Node: {node}",
                    f"Kind: {node_meta['kind']}",
                    (
                        f"Layer: {node_meta['layer']}"
                        if node_meta["kind"] not in {"input", "logits"}
                        else "Layer: N/A"
                    ),
                    f"Incident |score| sum: {incident_strength[node]:.4f}",
                ]
            )
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(color="white", width=1.2),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=node_hover,
            showlegend=False,
        )
    )

    tickvals = [0.0] + [float(layer + 1) for layer in range(max_layer + 1)] + [float(max_layer + 2)]
    ticktext = ["input"] + [f"L{layer}" for layer in range(max_layer + 1)] + ["logits"]

    fig.update_layout(
        title=f"Layered circuit graph for top {len(top_edges)} collapsed EAP edges",
        template="plotly_white",
        height=max(550, 32 * len(nodes)),
        hovermode="closest",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(
        title="Model depth",
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        title="Components within layer",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            "Attention heads are offset left within each layer; MLPs are offset right. "
            "Blue edges have positive signed attribution for the chosen metric, red edges negative."
        ),
    )
    return fig


def build_budget_curve_figure(budget_frame: pd.DataFrame) -> go.Figure:
    if budget_frame.empty:
        raise ValueError("budget_frame is empty. Run the greedy budget sweep before plotting.")

    frame = budget_frame.copy()
    if "budget_fraction" not in frame.columns:
        max_budget = float(frame["collapsed_edge_budget"].max())
        frame["budget_fraction"] = (
            frame["collapsed_edge_budget"] / max_budget if max_budget else 0.0
        )
    frame = frame.sort_values("budget_fraction").reset_index(drop=True)
    customdata = frame[["collapsed_edge_budget", "validation_examples"]].to_numpy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=frame["budget_fraction"],
            y=frame["faithfulness_mean"],
            mode="lines+markers",
            name="Faithfulness",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=7),
            error_y=dict(type="data", array=frame["faithfulness_std"], visible=True),
            customdata=customdata,
            hovertemplate=(
                "Budget share=%{x:.2%}<br>"
                "Collapsed edge budget=%{customdata[0]}<br>"
                "Faithfulness=%{y:.4f}<br>"
                "Validation examples=%{customdata[1]}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["budget_fraction"],
            y=frame["accuracy_mean"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
            marker=dict(size=7),
            error_y=dict(type="data", array=frame["accuracy_std"], visible=True),
            customdata=customdata,
            hovertemplate=(
                "Budget share=%{x:.2%}<br>"
                "Collapsed edge budget=%{customdata[0]}<br>"
                "Accuracy=%{y:.4f}<br>"
                "Validation examples=%{customdata[1]}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Faithfulness and accuracy across relative greedy edge budgets",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.update_xaxes(
        title="Collapsed edge budget (% of ranked model edges)",
        tickmode="array",
        tickvals=list(RELATIVE_BUDGET_TICKVALS),
        ticktext=list(RELATIVE_BUDGET_TICKTEXT),
    )
    fig.update_yaxes(title="Faithfulness", secondary_y=False)
    fig.update_yaxes(title="Accuracy", secondary_y=True, range=[0.0, 1.0])
    return fig


def build_attention_head_score_figure(node_frame: pd.DataFrame) -> go.Figure | None:
    attn_rows: list[dict[str, Any]] = []
    for row in node_frame.itertuples(index=False):
        meta = eap_node_metadata(str(row.node))
        if meta["kind"] != "attn":
            continue
        attn_rows.append(
            {
                "node": str(row.node),
                "layer": int(meta["layer"]),
                "head": int(meta["head"]),
                "induced_score": float(row.induced_score),
                "rank": int(row.rank),
            }
        )
    if not attn_rows:
        return None

    frame = pd.DataFrame(attn_rows)
    heatmap = (
        frame.pivot(index="layer", columns="head", values="induced_score")
        .sort_index()
        .sort_index(axis=1)
        .fillna(0.0)
    )
    customdata = np.empty((heatmap.shape[0], heatmap.shape[1]), dtype=object)
    for row_idx, layer in enumerate(heatmap.index.tolist()):
        for col_idx, head in enumerate(heatmap.columns.tolist()):
            match = frame[(frame["layer"] == layer) & (frame["head"] == head)]
            rank = int(match["rank"].iloc[0]) if not match.empty else None
            customdata[row_idx, col_idx] = rank

    fig = px.imshow(
        heatmap.to_numpy(),
        x=[f"H{head}" for head in heatmap.columns.tolist()],
        y=[f"L{layer}" for layer in heatmap.index.tolist()],
        labels={"x": "Attention head", "y": "Layer", "color": "Induced score"},
        color_continuous_scale="Blues",
        aspect="auto",
        title="Attention-head induced scores from EAP node rankings",
    )
    fig.update_traces(
        customdata=customdata,
        hovertemplate=(
            "Layer=%{y}<br>"
            "Head=%{x}<br>"
            "Induced score=%{z:.4f}<br>"
            "Node rank=%{customdata}<extra></extra>"
        ),
    )
    fig.update_layout(template="plotly_white", margin=dict(l=50, r=50, t=80, b=50))
    return fig


def build_mlp_layer_score_figure(node_frame: pd.DataFrame) -> go.Figure | None:
    mlp_rows: list[dict[str, Any]] = []
    for row in node_frame.itertuples(index=False):
        meta = eap_node_metadata(str(row.node))
        if meta["kind"] != "mlp":
            continue
        mlp_rows.append(
            {
                "node": str(row.node),
                "layer": int(meta["layer"]),
                "induced_score": float(row.induced_score),
                "rank": int(row.rank),
            }
        )
    if not mlp_rows:
        return None

    frame = pd.DataFrame(mlp_rows).sort_values("layer").reset_index(drop=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=[f"L{layer}" for layer in frame["layer"].tolist()],
                y=frame["induced_score"].tolist(),
                customdata=frame[["node", "rank"]].to_numpy(),
                marker=dict(color="#ff7f0e"),
                hovertemplate=(
                    "Layer=%{x}<br>"
                    "Node=%{customdata[0]}<br>"
                    "Induced score=%{y:.4f}<br>"
                    "Node rank=%{customdata[1]}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="MLP induced scores from EAP node rankings",
        template="plotly_white",
        xaxis_title="Layer",
        yaxis_title="Induced score",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    return fig


def build_layer_flow_figure(edge_frame: pd.DataFrame) -> go.Figure | None:
    if edge_frame.empty:
        return None

    flow_rows: list[dict[str, Any]] = []
    for row in edge_frame.itertuples(index=False):
        parent_meta = eap_node_metadata(str(row.parent))
        child_meta = eap_node_metadata(str(row.child))
        if parent_meta["kind"] in {"input", "logits"} or child_meta["kind"] in {"input", "logits"}:
            continue
        flow_rows.append(
            {
                "parent_layer": int(parent_meta["layer"]),
                "child_layer": int(child_meta["layer"]),
                "abs_score": float(row.abs_score),
            }
        )
    if not flow_rows:
        return None

    frame = (
        pd.DataFrame(flow_rows)
        .groupby(["parent_layer", "child_layer"], as_index=False)["abs_score"]
        .sum()
    )
    heatmap = (
        frame.pivot(index="parent_layer", columns="child_layer", values="abs_score")
        .sort_index()
        .sort_index(axis=1)
        .fillna(0.0)
    )
    fig = px.imshow(
        heatmap.to_numpy(),
        x=[f"L{layer}" for layer in heatmap.columns.tolist()],
        y=[f"L{layer}" for layer in heatmap.index.tolist()],
        labels={
            "x": "Child layer",
            "y": "Parent layer",
            "color": "Summed |edge score|",
        },
        color_continuous_scale="Blues",
        aspect="auto",
        title="Layer-to-layer EAP edge mass",
    )
    fig.update_traces(
        hovertemplate=(
            "Parent layer=%{y}<br>"
            "Child layer=%{x}<br>"
            "Summed |edge score|=%{z:.4f}<extra></extra>"
        )
    )
    fig.update_layout(template="plotly_white", margin=dict(l=50, r=50, t=80, b=50))
    return fig


def write_plotly_figure_bundle(
    fig: go.Figure,
    html_path: Path,
    image_dir: Path,
    image_stem: str,
    export_static_image: bool = True,
) -> dict[str, str]:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path)

    artifact_paths = {"html": str(html_path)}
    if not export_static_image:
        return artifact_paths

    image_path = image_dir / f"{image_stem}.png"
    try:
        fig.write_image(str(image_path))
    except Exception as exc:
        print(
            "WARNING: Could not export static image "
            f"to {image_path}: {type(exc).__name__}: {exc}"
        )
    else:
        artifact_paths["png"] = str(image_path)
    return artifact_paths


def save_eap_visualizations(
    project_root: Path,
    output_dir: Path,
    edge_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    budget_frame: pd.DataFrame,
    day: str,
    top_k_edges: int | None = None,
    export_static_images: bool = False,
) -> dict[str, dict[str, str]]:
    image_dir = resolve_image_output_dir(project_root, output_dir)
    top_k = min(top_k_edges or 40, len(edge_frame)) if len(edge_frame) else 0
    artifacts: dict[str, dict[str, str]] = {}

    if top_k > 0:
        layered_fig = build_layered_circuit_figure(edge_frame, top_k=top_k)
        artifacts["layered_circuit"] = write_plotly_figure_bundle(
            layered_fig,
            output_dir / f"layered_circuit_top_{top_k}_edges_{day}.html",
            image_dir,
            f"layered_circuit_top_{top_k}_edges_{day}",
            export_static_image=export_static_images,
        )

    if not budget_frame.empty:
        budget_fig = build_budget_curve_figure(budget_frame)
        artifacts["budget_curve"] = write_plotly_figure_bundle(
            budget_fig,
            output_dir / f"budget_sweep_curve_{day}.html",
            image_dir,
            f"budget_sweep_curve_{day}",
            export_static_image=export_static_images,
        )

    attention_fig = build_attention_head_score_figure(node_frame)
    if attention_fig is not None:
        artifacts["attention_head_scores"] = write_plotly_figure_bundle(
            attention_fig,
            output_dir / f"attention_head_induced_scores_{day}.html",
            image_dir,
            f"attention_head_induced_scores_{day}",
            export_static_image=export_static_images,
        )

    mlp_fig = build_mlp_layer_score_figure(node_frame)
    if mlp_fig is not None:
        artifacts["mlp_layer_scores"] = write_plotly_figure_bundle(
            mlp_fig,
            output_dir / f"mlp_layer_induced_scores_{day}.html",
            image_dir,
            f"mlp_layer_induced_scores_{day}",
            export_static_image=export_static_images,
        )

    flow_fig = build_layer_flow_figure(edge_frame)
    if flow_fig is not None:
        artifacts["layer_flow"] = write_plotly_figure_bundle(
            flow_fig,
            output_dir / f"layer_flow_abs_scores_{day}.html",
            image_dir,
            f"layer_flow_abs_scores_{day}",
            export_static_image=export_static_images,
        )

    return artifacts


def variant_selection_summary(
    variant_id: str,
    pipeline: str,
    threshold: int | None,
    budget_results: Sequence[BudgetEvaluation],
) -> VariantSelectionSummary | None:
    if not budget_results:
        return None

    supported_budgets = [result.budget for result in budget_results]
    mean_faithfulness = float(np.mean([result.faithfulness_mean for result in budget_results]))
    mean_induced_nodes = float(np.mean([result.induced_node_count for result in budget_results]))
    return VariantSelectionSummary(
        variant_id=variant_id,
        pipeline=pipeline,
        threshold=threshold,
        supported_budgets=supported_budgets,
        mean_faithfulness=mean_faithfulness,
        mean_induced_nodes=mean_induced_nodes,
    )


def choose_final_variant(
    summaries: Sequence[VariantSelectionSummary],
    tolerance: float,
) -> dict[str, Any] | None:
    if not summaries:
        return None

    def key(summary: VariantSelectionSummary) -> tuple[float, float, int]:
        return (
            summary.mean_faithfulness,
            -summary.mean_induced_nodes,
            len(summary.supported_budgets),
        )

    best = summaries[0]
    for candidate in summaries[1:]:
        if candidate.mean_faithfulness > best.mean_faithfulness + tolerance:
            best = candidate
            continue
        if abs(candidate.mean_faithfulness - best.mean_faithfulness) <= tolerance:
            if candidate.mean_induced_nodes + tolerance < best.mean_induced_nodes:
                best = candidate
                continue
            if abs(candidate.mean_induced_nodes - best.mean_induced_nodes) <= tolerance:
                if len(candidate.supported_budgets) > len(best.supported_budgets):
                    best = candidate

    return asdict(best)


def is_nonincreasing(values: Sequence[int]) -> bool:
    return all(next_value <= value for value, next_value in zip(values, values[1:]))


def run_pipeline_a(
    project_root: Path,
    output_dir: Path,
    model: HookedTransformer,
    discovery_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    metric: Callable[..., torch.Tensor],
    config: ComparisonConfig,
) -> dict[str, Any]:
    full_graph = build_graph(model)
    discovery_loader = make_dataloader(
        discovery_df,
        batch_size=config.attribution_batch_size,
        shuffle=False,
    )
    validation_loader = make_dataloader(
        validation_df,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
    )
    scored_graph = attribute_graph(
        model=model,
        graph=full_graph,
        dataloader=discovery_loader,
        metric=metric,
        ig_steps=config.ig_steps,
    )
    ranked_edges = collapsed_edge_groups(scored_graph)
    ranked_nodes = induced_node_ranking(ranked_edges)
    save_rankings(output_dir, "pipeline_a_full_graph", ranked_edges, ranked_nodes)

    budget_results: list[BudgetEvaluation] = []
    for budget in config.budgets:
        if budget > len(ranked_edges):
            continue
        budget_results.append(
            evaluate_budget(
                model=model,
                scored_graph=scored_graph,
                ranked_collapsed_edges=ranked_edges,
                validation_loader=validation_loader,
                metric=metric,
                budget=budget,
            )
        )

    return {
        "variant_id": "pipeline_a_full_graph",
        "pipeline": "A",
        "threshold": None,
        "full_graph_collapsed_edge_count": len(ranked_edges),
        "full_graph_expanded_edge_count": int(scored_graph.real_edge_mask.sum().item()),
        "ranked_edge_count": len(ranked_edges),
        "ranked_node_count": len(ranked_nodes),
        "budget_results": [asdict(result) for result in budget_results],
    }


def run_pipeline_b(
    project_root: Path,
    output_dir: Path,
    model: HookedTransformer,
    tokenizer,
    discovery_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    config: ComparisonConfig,
) -> dict[str, Any]:
    from transformer_lens import patching as tl_patching

    patch_metric_factory = make_avg_logit_difference_recovery_metric(
        animate_ids_tensor,
        inanimate_ids_tensor,
    )
    eap_metric = make_eap_normalized_recovery_metric(
        animate_ids_tensor,
        inanimate_ids_tensor,
    )

    residual_path = run_and_save_experiment(
        project_root=project_root,
        model=model,
        experiment_name="Residual_Stream_Patching",
        df=discovery_df,
        tokenizer=tokenizer,
        patching_func=tl_patching.get_act_patch_resid_pre,
        metric_factory=patch_metric_factory,
        filter_str="hook_resid_pre",
        batch_size=config.patch_batch_size,
        safety_checks=True,
        output_dir=output_dir,
    )
    residual_rows = heatmap_to_score_rows(torch.load(residual_path))
    positive_residual_rows = residual_rows[residual_rows["score"] > 0].copy()

    full_graph = build_graph(model)
    validation_loader = make_dataloader(
        validation_df,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
    )
    discovery_loader = make_dataloader(
        discovery_df,
        batch_size=config.attribution_batch_size,
        shuffle=False,
    )

    threshold_variants: list[dict[str, Any]] = []
    for threshold in config.thresholds:
        retained_residual_rows = select_positive_rows_at_percentile(
            positive_residual_rows,
            threshold,
        )
        retained_sites = patch_rows_to_sites(retained_residual_rows)

        if retained_sites:
            mlp_patch_path = run_and_save_experiment(
                project_root=project_root,
                model=model,
                experiment_name=f"Hybrid_MLP_Patching_t{threshold}",
                df=discovery_df,
                tokenizer=tokenizer,
                patching_func=make_multi_site_patching_func(retained_sites, component="mlp_out"),
                metric_factory=patch_metric_factory,
                filter_str="hook_mlp_out",
                batch_size=config.patch_batch_size,
                output_dir=output_dir,
            )
            attn_patch_path = run_and_save_experiment(
                project_root=project_root,
                model=model,
                experiment_name=f"Hybrid_Attn_Patching_t{threshold}",
                df=discovery_df,
                tokenizer=tokenizer,
                patching_func=make_multi_site_patching_func(retained_sites, component="attn_out"),
                metric_factory=patch_metric_factory,
                filter_str="hook_attn_out",
                batch_size=config.patch_batch_size,
                output_dir=output_dir,
            )
            mlp_rows = heatmap_to_score_rows(torch.load(mlp_patch_path)).assign(component="mlp")
            attn_rows = heatmap_to_score_rows(torch.load(attn_patch_path)).assign(component="attn")
            module_rows = pd.concat([mlp_rows, attn_rows], ignore_index=True)
            retained_module_rows = select_positive_rows_at_percentile(
                module_rows[module_rows["score"] > 0].copy(),
                threshold,
            )
        else:
            mlp_patch_path = None
            attn_patch_path = None
            retained_module_rows = pd.DataFrame(
                columns=["layer", "token_position", "token_position_from_end", "token", "score", "component"]
            )

        retained_mlp_layers = {
            int(layer)
            for layer in retained_module_rows.loc[
                retained_module_rows["component"] == "mlp", "layer"
            ].tolist()
        }
        retained_attn_layers = {
            int(layer)
            for layer in retained_module_rows.loc[
                retained_module_rows["component"] == "attn", "layer"
            ].tolist()
        }

        retained_nodes = {f"m{layer}" for layer in retained_mlp_layers}
        for layer in retained_attn_layers:
            retained_nodes.update(
                f"a{layer}.h{head}" for head in range(model.cfg.n_heads)
            )

        reduced_graph, allowed_edge_names = build_reduced_graph(full_graph, retained_nodes)
        ranked_edges: list[dict[str, Any]] = []
        ranked_nodes: list[dict[str, Any]] = []
        budget_results: list[BudgetEvaluation] = []

        if allowed_edge_names:
            scored_graph = attribute_graph(
                model=model,
                graph=reduced_graph,
                dataloader=discovery_loader,
                metric=eap_metric,
                ig_steps=config.ig_steps,
            )
            ranked_edges = collapsed_edge_groups(scored_graph)
            ranked_nodes = induced_node_ranking(ranked_edges)
            save_rankings(
                output_dir,
                f"pipeline_b_threshold_{threshold}",
                ranked_edges,
                ranked_nodes,
            )

            for budget in config.budgets:
                if budget > len(ranked_edges):
                    continue
                budget_results.append(
                    evaluate_budget(
                        model=model,
                        scored_graph=scored_graph,
                        ranked_collapsed_edges=ranked_edges,
                        validation_loader=validation_loader,
                        metric=eap_metric,
                        budget=budget,
                    )
                )
        else:
            scored_graph = reduced_graph

        threshold_variants.append(
            {
                "variant_id": f"pipeline_b_threshold_{threshold}",
                "pipeline": "B",
                "threshold": threshold,
                "residual_patch_file": str(residual_path),
                "module_patch_files": {
                    "mlp": str(mlp_patch_path) if mlp_patch_path is not None else None,
                    "attn": str(attn_patch_path) if attn_patch_path is not None else None,
                },
                "retained_residual_site_count": int(len(retained_residual_rows)),
                "retained_module_count": int(len(retained_module_rows)),
                "retained_mlp_layers": sorted(retained_mlp_layers),
                "retained_attention_block_layers": sorted(retained_attn_layers),
                "retained_module_rows": retained_module_rows.to_dict("records"),
                "retained_nodes": sorted(retained_nodes),
                "reduced_graph_collapsed_edge_count": collapsed_edge_count_from_mask(scored_graph),
                "reduced_graph_expanded_edge_count": int(scored_graph.real_edge_mask.sum().item()),
                "ranked_edge_count": len(ranked_edges),
                "ranked_node_count": len(ranked_nodes),
                "budget_results": [asdict(result) for result in budget_results],
            }
        )

    return {
        "residual_patch_file": str(residual_path),
        "variants": threshold_variants,
    }


def load_model_context(
    project_root: Path,
    model_name: str,
    target_filter_model_names: Sequence[str] | None = None,
    target_source: str | Path | None = None,
) -> dict[str, Any]:
    resolved_model_name = canonical_model_name(model_name)
    model = load_model(model_name)
    tokenizer = model.tokenizer

    target_filter_models = unique_model_names(
        target_filter_model_names
        if target_filter_model_names is not None
        else DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS
    )
    animate_words, inanimate_words, target_filter_summary, target_filter_path = (
        load_or_filter_targets_for_models(
            project_root,
            target_filter_models,
            target_tokenizer=tokenizer,
            target_source=target_source,
        )
    )
    animate_ids_tensor = create_target_tensor(
        animate_words,
        tokenizer,
        model.cfg.device,
    )
    inanimate_ids_tensor = create_target_tensor(
        inanimate_words,
        tokenizer,
        model.cfg.device,
    )
    target_tokenization_diagnostics = build_target_tokenization_diagnostics(
        animate_words,
        inanimate_words,
        tokenizer,
    )
    verify_target_tensors(animate_words, animate_ids_tensor, tokenizer)
    verify_target_tensors(inanimate_words, inanimate_ids_tensor, tokenizer)

    return {
        "requested_model_name": model_name,
        "model_name": resolved_model_name,
        "model": model,
        "tokenizer": tokenizer,
        "animate_words": animate_words,
        "inanimate_words": inanimate_words,
        "animate_ids_tensor": animate_ids_tensor,
        "inanimate_ids_tensor": inanimate_ids_tensor,
        "target_tokenization_diagnostics": target_tokenization_diagnostics,
        "target_filter_model_names": target_filter_models,
        "target_filter_summary": target_filter_summary,
        "target_filter_path": str(target_filter_path) if target_filter_path is not None else None,
        "target_source": str(target_source or DEFAULT_TARGET_SOURCE),
        "target_source_path": str(resolve_target_source_path(project_root, target_source)),
    }


def prepare_filtered_model_inputs(
    project_root: Path,
    model_name: str,
    dataset_filter_model_name: str,
    metric_batch_size: int,
    seed: int,
    dataset_filter_path: Path | str | None = None,
    refresh_dataset_filter: bool = False,
    cache_dataset_filter: bool = True,
    max_filter_examples: int | None = None,
    target_filter_policy: str = "model_success",
    target_source: str | Path | None = None,
) -> dict[str, Any]:
    resolved_model_name = canonical_model_name(model_name)
    resolved_dataset_filter_model_name = canonical_model_name(dataset_filter_model_name)
    print(
        f"Preparing source success pool with {resolved_dataset_filter_model_name}; "
        f"target model is {resolved_model_name}."
    )
    source_success_df = load_or_create_model_success_dataset(
        project_root=project_root,
        model_name=resolved_dataset_filter_model_name,
        batch_size=metric_batch_size,
        cache_path=dataset_filter_path,
        refresh=refresh_dataset_filter,
        cache=cache_dataset_filter or dataset_filter_path is not None,
        max_examples=max_filter_examples,
        seed=seed,
        target_source=target_source,
    )

    print(f"Loading target model {resolved_model_name}.")
    common_filter_model_names = unique_model_names(DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS)
    context = load_model_context(
        project_root,
        model_name,
        target_filter_model_names=common_filter_model_names,
        target_source=target_source,
    )
    raw_dataset_with_metadata = attach_pair_metadata(
        load_common_tokenized_pairs(project_root, common_filter_model_names),
        project_root,
    )
    raw_tokenization_diagnostics = token_alignment_diagnostics(
        raw_dataset_with_metadata,
        tokenizer=context["tokenizer"],
    )
    if (
        resolved_dataset_filter_model_name == resolved_model_name
        and target_filter_policy == "model_success"
        and max_filter_examples is None
    ):
        target_raw_scored_df = source_success_df.copy()
        target_scored_df = source_success_df.copy()
        filtered_df = source_success_df.copy()
    else:
        tokenization_source_df = load_common_tokenized_pairs(project_root, common_filter_model_names)
        target_raw_scored_df = compute_model_scored_dataset(
            project_root=project_root,
            model=context["model"],
            tokenizer=context["tokenizer"],
            animate_ids_tensor=context["animate_ids_tensor"],
            inanimate_ids_tensor=context["inanimate_ids_tensor"],
            batch_size=metric_batch_size,
            source_df=tokenization_source_df,
            seed=seed,
            max_examples=max_filter_examples,
        )
        target_scored_df = intersect_target_scores_with_source_success(
            target_scored_df=target_raw_scored_df,
            source_success_df=source_success_df,
            dataset_filter_model_name=resolved_dataset_filter_model_name,
        )
        filtered_df = apply_target_filter_policy(target_scored_df, target_filter_policy)

    return {
        "source_success_df": source_success_df,
        "target_raw_scored_df": target_raw_scored_df,
        "target_scored_df": target_scored_df,
        "filtered_df": filtered_df,
        "raw_tokenization_diagnostics": raw_tokenization_diagnostics,
        "target_raw_accuracy": task_accuracy_summary(target_raw_scored_df),
        "target_on_source_accuracy": task_accuracy_summary(target_scored_df),
        "source_success_cache_path": source_success_df.attrs.get("model_success_cache_path"),
        "source_success_cache_status": source_success_df.attrs.get("model_success_cache_status"),
        "requested_model_name": model_name,
        "model_name": resolved_model_name,
        "requested_dataset_filter_model_name": dataset_filter_model_name,
        "dataset_filter_model_name": resolved_dataset_filter_model_name,
        **context,
    }


def concept_site_key(hook_name: str) -> str:
    return hook_name.replace(".", "__")


def concept_hook_name(layer: int, hook_point: str) -> str:
    return f"blocks.{int(layer)}.{hook_point}"


def normalize_concept_pair_metadata(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in ("patient", "clean_verb", "corrupt_verb", "uid", "domain"):
        candidates = [
            candidate
            for candidate in (column, f"{column}_x", f"{column}_y")
            if candidate in normalized.columns
        ]
        if not candidates:
            continue
        series = normalized[candidates[0]].copy()
        for candidate in candidates[1:]:
            series = series.combine_first(normalized[candidate])
        normalized[column] = series
    return normalized


def add_concept_verb_positions(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    df = normalize_concept_pair_metadata(df)
    metadata_available = {"patient", "clean_verb", "corrupt_verb"}.issubset(df.columns)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row_idx, row in df.reset_index(drop=True).iterrows():
        details = pair_token_alignment_details(
            row,
            tokenizer,
            metadata_available=metadata_available,
        )
        if not details["pair_ok"]:
            failures.append(
                {
                    "row": int(row_idx),
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                    "clean_verb": details["clean_verb"],
                    "corrupt_verb": details["corrupt_verb"],
                    "clean_verb_span": details["clean_verb_span"],
                    "corrupt_verb_span": details["corrupt_verb_span"],
                    "clean_verb_error": details["clean_verb_error"],
                    "corrupt_verb_error": details["corrupt_verb_error"],
                }
            )
            continue

        clean_span = details["clean_verb_span"]
        corrupt_span = details["corrupt_verb_span"]
        assert clean_span is not None
        assert corrupt_span is not None
        if (clean_span[1] - clean_span[0]) != 1 or (corrupt_span[1] - corrupt_span[0]) != 1:
            failures.append(
                {
                    "row": int(row_idx),
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                    "clean_verb": details["clean_verb"],
                    "corrupt_verb": details["corrupt_verb"],
                    "clean_verb_span": clean_span,
                    "corrupt_verb_span": corrupt_span,
                    "clean_verb_error": "verb_not_single_token",
                    "corrupt_verb_error": "verb_not_single_token",
                }
            )
            continue

        item = row.to_dict()
        item["verb_token_position"] = int(clean_span[0] + 1)
        item["clean_verb_token_position"] = int(clean_span[0] + 1)
        item["corrupt_verb_token_position"] = int(corrupt_span[0] + 1)
        rows.append(item)

    if failures:
        preview = failures[:5]
        raise ValueError(
            "Concept extraction requires aligned single-token clean/corrupt verbs. "
            f"Found {len(failures)} invalid rows; first failures: {preview}"
        )
    if not rows:
        raise ValueError("No rows available after concept verb-position validation.")
    return pd.DataFrame(rows).reset_index(drop=True)


def split_concept_dataset_by_uid(
    df: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty concept dataset.")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be less than 1.")

    working = df.reset_index(drop=True).copy()
    split_keys: list[str] = []
    for _, row in working.iterrows():
        uid = row.get("uid")
        if uid is not None and not pd.isna(uid):
            split_keys.append(str(uid))
            continue
        split_keys.append(
            hashlib.sha256(
                f"{row['clean_prefix']} || {row['corrupt_prefix']}".encode("utf-8")
            ).hexdigest()
        )
    working["_concept_split_key"] = split_keys

    key_sizes = working.groupby("_concept_split_key").size().reset_index(name="count")
    rng = np.random.default_rng(seed)
    shuffled_keys = key_sizes["_concept_split_key"].to_numpy(copy=True)
    rng.shuffle(shuffled_keys)

    total = len(working)
    train_target = int(round(total * train_fraction))
    validation_target = int(round(total * validation_fraction))
    size_by_key = dict(zip(key_sizes["_concept_split_key"], key_sizes["count"]))
    split_by_key: dict[str, str] = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    for key in shuffled_keys:
        key_str = str(key)
        key_count = int(size_by_key[key_str])
        if counts["train"] < train_target:
            split_name = "train"
        elif counts["validation"] < validation_target:
            split_name = "validation"
        else:
            split_name = "test"
        split_by_key[key_str] = split_name
        counts[split_name] += key_count

    working["_concept_split"] = [split_by_key[key] for key in working["_concept_split_key"]]
    splits = {
        split_name: (
            working.loc[working["_concept_split"] == split_name]
            .drop(columns=["_concept_split", "_concept_split_key"])
            .reset_index(drop=True)
            .copy()
        )
        for split_name in ("train", "validation", "test")
    }
    empty = [split_name for split_name, split_df in splits.items() if split_df.empty]
    if empty:
        raise ValueError(
            f"UID split produced empty split(s): {empty}. "
            "Use more filtered examples or reduce max_filter_examples."
        )
    return splits


def extract_animacy_concept_vectors(
    model: HookedTransformer,
    train_df: pd.DataFrame,
    hook_points: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    hook_names = [
        concept_hook_name(layer, hook_point)
        for layer in range(int(model.cfg.n_layers))
        for hook_point in hook_points
    ]
    sums_clean = {
        hook_name: torch.zeros(int(model.cfg.d_model), dtype=torch.float64)
        for hook_name in hook_names
    }
    sums_corrupt = {
        hook_name: torch.zeros(int(model.cfg.d_model), dtype=torch.float64)
        for hook_name in hook_names
    }
    count = 0
    estimated_batches = sum(
        math.ceil(len(group) / batch_size)
        for _, group in train_df.groupby("seq_len")
    )

    for clean_tokens, corrupt_tokens, batch_df in tqdm(
        generate_exact_length_batches(train_df, model, batch_size, model.cfg.device),
        total=estimated_batches,
        desc="Extracting concept activations",
    ):
        positions = torch.tensor(
            batch_df["verb_token_position"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=model.cfg.device,
        )
        batch_indices = torch.arange(clean_tokens.shape[0], device=model.cfg.device)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_names)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=hook_names)
        for hook_name in hook_names:
            clean_selected = clean_cache[hook_name][batch_indices, positions, :]
            corrupt_selected = corrupt_cache[hook_name][batch_indices, positions, :]
            sums_clean[hook_name] += clean_selected.detach().float().sum(dim=0).cpu().double()
            sums_corrupt[hook_name] += corrupt_selected.detach().float().sum(dim=0).cpu().double()
        count += int(clean_tokens.shape[0])
        del clean_cache, corrupt_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if count == 0:
        raise ValueError("No training examples were available for concept extraction.")

    vectors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for layer in range(int(model.cfg.n_layers)):
        for hook_point in hook_points:
            hook_name = concept_hook_name(layer, hook_point)
            clean_mean = sums_clean[hook_name] / count
            corrupt_mean = sums_corrupt[hook_name] / count
            vector = (clean_mean - corrupt_mean).float()
            vectors[hook_name] = vector
            rows.append(
                {
                    "layer": int(layer),
                    "hook_point": hook_point,
                    "hook_name": hook_name,
                    "site_key": concept_site_key(hook_name),
                    "train_examples": int(count),
                    "clean_mean_norm": float(clean_mean.float().norm().item()),
                    "corrupt_mean_norm": float(corrupt_mean.float().norm().item()),
                    "concept_norm": float(vector.norm().item()),
                }
            )
    return vectors, pd.DataFrame(rows)


def make_concept_steering_hook(
    positions: Sequence[int],
    vector: torch.Tensor,
    alpha: float,
):
    position_tensor: torch.Tensor | None = None
    batch_indices: torch.Tensor | None = None

    def hook_fn(activation: torch.Tensor, hook):
        nonlocal position_tensor, batch_indices
        del hook
        if position_tensor is None or position_tensor.device != activation.device:
            position_tensor = torch.tensor(positions, dtype=torch.long, device=activation.device)
            batch_indices = torch.arange(len(positions), dtype=torch.long, device=activation.device)
        assert batch_indices is not None
        intervention = vector.to(device=activation.device, dtype=activation.dtype) * float(alpha)
        patched = activation.clone()
        patched[batch_indices, position_tensor, :] = (
            patched[batch_indices, position_tensor, :] + intervention
        )
        return patched

    return hook_fn


def concept_steering_vector(raw_vector: torch.Tensor, normalize: bool) -> torch.Tensor:
    vector = raw_vector.detach().float().cpu()
    if not normalize:
        return vector
    norm = vector.norm()
    if float(norm.item()) <= 0:
        return vector
    return vector / norm


def random_control_vector(
    reference_vector: torch.Tensor,
    seed: int,
    repeat_index: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1009 * int(repeat_index + 1))
    random_vector = torch.randn(reference_vector.shape, generator=generator, dtype=torch.float32)
    random_norm = random_vector.norm()
    if float(random_norm.item()) <= 0:
        return random_vector
    reference_norm = reference_vector.detach().float().cpu().norm()
    return random_vector / random_norm * reference_norm


def summarize_concept_steering_rows(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "example_count": 0,
            "clean_signed_effect_mean": 0.0,
            "corrupt_signed_effect_mean": 0.0,
            "signed_effect_mean": 0.0,
            "signed_effect_std": 0.0,
            "clean_flip_rate": 0.0,
            "corrupt_flip_rate": 0.0,
        }
    signed_effect = rows["example_signed_effect"]
    return {
        "example_count": int(len(rows)),
        "clean_before_mean": float(rows["clean_metric_before"].mean()),
        "clean_after_mean": float(rows["clean_metric_after"].mean()),
        "corrupt_before_mean": float(rows["corrupt_metric_before"].mean()),
        "corrupt_after_mean": float(rows["corrupt_metric_after"].mean()),
        "clean_signed_effect_mean": float(rows["clean_signed_effect"].mean()),
        "corrupt_signed_effect_mean": float(rows["corrupt_signed_effect"].mean()),
        "signed_effect_mean": float(signed_effect.mean()),
        "signed_effect_std": float(signed_effect.std(ddof=0)) if len(rows) > 1 else 0.0,
        "clean_flip_rate": float(rows["clean_flipped_below_zero"].mean()),
        "corrupt_flip_rate": float(rows["corrupt_flipped_above_zero"].mean()),
    }


def evaluate_concept_steering(
    model: HookedTransformer,
    df: pd.DataFrame,
    hook_name: str,
    concept_vector: torch.Tensor,
    alpha: float,
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
    batch_size: int,
    split_name: str,
    return_rows: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    rows: list[dict[str, Any]] = []
    estimated_batches = sum(
        math.ceil(len(group) / batch_size)
        for _, group in df.groupby("seq_len")
    )
    animate_ids = animate_ids_tensor.to(model.cfg.device)
    inanimate_ids = inanimate_ids_tensor.to(model.cfg.device)

    for clean_tokens, corrupt_tokens, batch_df in tqdm(
        generate_exact_length_batches(df, model, batch_size, model.cfg.device),
        total=estimated_batches,
        desc=f"Evaluating {split_name} {hook_name} alpha={alpha:g}",
        leave=False,
    ):
        positions = batch_df["verb_token_position"].astype(int).tolist()
        with torch.no_grad():
            clean_before_logits = model(clean_tokens)
            corrupt_before_logits = model(corrupt_tokens)
            clean_after_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (
                        hook_name,
                        make_concept_steering_hook(
                            positions,
                            concept_vector,
                            -float(alpha),
                        ),
                    )
                ],
            )
            corrupt_after_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[
                    (
                        hook_name,
                        make_concept_steering_hook(
                            positions,
                            concept_vector,
                            float(alpha),
                        ),
                    )
                ],
            )

        clean_before = average_logit_difference_from_final_logits(
            clean_before_logits[:, -1, :],
            animate_ids,
            inanimate_ids,
        )
        corrupt_before = average_logit_difference_from_final_logits(
            corrupt_before_logits[:, -1, :],
            animate_ids,
            inanimate_ids,
        )
        clean_after = average_logit_difference_from_final_logits(
            clean_after_logits[:, -1, :],
            animate_ids,
            inanimate_ids,
        )
        corrupt_after = average_logit_difference_from_final_logits(
            corrupt_after_logits[:, -1, :],
            animate_ids,
            inanimate_ids,
        )

        for row, cb, ca, xb, xa in zip(
            batch_df.to_dict("records"),
            clean_before.detach().cpu().tolist(),
            clean_after.detach().cpu().tolist(),
            corrupt_before.detach().cpu().tolist(),
            corrupt_after.detach().cpu().tolist(),
        ):
            clean_signed = float(cb - ca)
            corrupt_signed = float(xa - xb)
            rows.append(
                {
                    "split": split_name,
                    "uid": row.get("uid"),
                    "clean_prefix": row["clean_prefix"],
                    "corrupt_prefix": row["corrupt_prefix"],
                    "patient": row.get("patient"),
                    "clean_verb": row.get("clean_verb"),
                    "corrupt_verb": row.get("corrupt_verb"),
                    "domain": row.get("domain"),
                    "hook_name": hook_name,
                    "alpha": float(alpha),
                    "clean_metric_before": float(cb),
                    "clean_metric_after": float(ca),
                    "corrupt_metric_before": float(xb),
                    "corrupt_metric_after": float(xa),
                    "clean_signed_effect": clean_signed,
                    "corrupt_signed_effect": corrupt_signed,
                    "example_signed_effect": float((clean_signed + corrupt_signed) / 2.0),
                    "clean_flipped_below_zero": bool(cb > 0 and ca < 0),
                    "corrupt_flipped_above_zero": bool(xb < 0 and xa > 0),
                }
            )
        del clean_before_logits, corrupt_before_logits, clean_after_logits, corrupt_after_logits

    row_frame = pd.DataFrame(rows)
    summary = summarize_concept_steering_rows(row_frame)
    return summary, (row_frame if return_rows else None)


def select_conservative_concept_site(
    validation_sweep: pd.DataFrame,
    effect_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if validation_sweep.empty:
        raise ValueError("Cannot select a concept site from an empty validation sweep.")
    if not 0 < effect_fraction <= 1:
        raise ValueError("selection_effect_fraction must be in (0, 1].")

    sweep = validation_sweep.copy()
    sweep["abs_alpha"] = sweep["alpha"].abs()
    best_effect = float(sweep["signed_effect_mean"].max())
    threshold = best_effect * float(effect_fraction) if best_effect > 0 else best_effect
    sweep["selection_threshold"] = threshold
    sweep["selection_eligible"] = sweep["signed_effect_mean"] >= threshold
    eligible = sweep.loc[sweep["selection_eligible"]].copy()
    if eligible.empty:
        eligible = sweep.copy()
    selected = eligible.sort_values(
        ["abs_alpha", "signed_effect_mean", "clean_signed_effect_mean", "corrupt_signed_effect_mean"],
        ascending=[True, False, False, False],
    ).iloc[0].to_dict()
    sweep["selected"] = (
        (sweep["hook_name"] == selected["hook_name"])
        & (sweep["alpha"] == selected["alpha"])
        & (sweep["signed_effect_mean"] == selected["signed_effect_mean"])
    )
    return selected, sweep.sort_values(
        ["signed_effect_mean", "clean_signed_effect_mean", "corrupt_signed_effect_mean"],
        ascending=[False, False, False],
    )


def run_concept_extraction_experiment(
    config: ConceptExtractionConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    if not config.alpha_grid:
        raise ValueError("alpha_grid must contain at least one value.")
    if config.random_control_repeats < 0:
        raise ValueError("random_control_repeats must be non-negative.")
    project_root = resolve_animacy_circuit_root(start)
    prepared = prepare_filtered_model_inputs(
        project_root=project_root,
        model_name=config.model_name,
        dataset_filter_model_name=config.dataset_filter_model_name,
        metric_batch_size=config.filter_batch_size,
        seed=config.seed,
        dataset_filter_path=config.dataset_filter_path,
        refresh_dataset_filter=config.refresh_dataset_filter,
        cache_dataset_filter=config.cache_dataset_filter,
        max_filter_examples=config.max_filter_examples,
        target_filter_policy=config.target_filter_policy,
        target_source=config.target_source,
    )
    model = prepared["model"]
    tokenizer = prepared["tokenizer"]
    filtered_df = prepared["filtered_df"].copy()
    if filtered_df.empty:
        raise ValueError("No filtered examples are available for concept extraction.")
    filtered_df = maybe_limit_examples(filtered_df, config.max_concept_examples, config.seed)

    filtered_df = add_sequence_lengths(filtered_df, model)
    filtered_df = add_concept_verb_positions(filtered_df, tokenizer)
    splits = split_concept_dataset_by_uid(filtered_df, seed=config.seed)

    output_dir = concept_extraction_dir(
        project_root,
        prepared["model_name"],
        config.output_day,
    )
    run_tag = timestamp_tag()
    split_paths: dict[str, str] = {}
    for split_name, split_df in splits.items():
        path = output_dir / f"{split_name}_split_{run_tag}.csv"
        save_csv(split_df, path, index=False)
        split_paths[split_name] = str(path)

    vectors, site_summary = extract_animacy_concept_vectors(
        model=model,
        train_df=splits["train"],
        hook_points=config.hook_points,
        batch_size=config.extraction_batch_size,
    )
    site_summary_path = output_dir / f"concept_site_summary_{run_tag}.csv"
    save_csv(site_summary, site_summary_path, index=False)

    vectors_path = output_dir / f"concept_vectors_{run_tag}.pt"
    save_torch(
        {
            "vectors": vectors,
            "site_summary": site_summary.to_dict("records"),
            "config": asdict(config),
            "model_name": prepared["model_name"],
            "dataset_filter_model_name": prepared["dataset_filter_model_name"],
        },
        vectors_path,
    )

    validation_rows: list[dict[str, Any]] = []
    site_meta = {
        row["hook_name"]: row
        for row in site_summary.to_dict("records")
    }
    validation_total = len(vectors) * len(config.alpha_grid)
    with tqdm(
        total=validation_total,
        desc="Validation steering sweep",
        unit="site-alpha",
    ) as validation_progress:
        for hook_name, concept_vector in vectors.items():
            steering_vector = concept_steering_vector(
                concept_vector,
                normalize=config.normalize_concept_vector,
            )
            for alpha in config.alpha_grid:
                validation_progress.set_postfix_str(
                    f"{hook_name} alpha={float(alpha):g}",
                    refresh=False,
                )
                summary, _ = evaluate_concept_steering(
                    model=model,
                    df=splits["validation"],
                    hook_name=hook_name,
                    concept_vector=steering_vector,
                    alpha=float(alpha),
                    animate_ids_tensor=prepared["animate_ids_tensor"],
                    inanimate_ids_tensor=prepared["inanimate_ids_tensor"],
                    batch_size=config.steering_batch_size,
                    split_name="validation",
                    return_rows=False,
                )
                validation_rows.append(
                    {
                        **site_meta[hook_name],
                        "alpha": float(alpha),
                        "steering_vector_norm": float(steering_vector.norm().item()),
                        "normalize_concept_vector": bool(config.normalize_concept_vector),
                        **summary,
                    }
                )
                validation_progress.update(1)
    selected, validation_sweep = select_conservative_concept_site(
        pd.DataFrame(validation_rows),
        effect_fraction=config.selection_effect_fraction,
    )
    validation_sweep_path = output_dir / f"validation_sweep_{run_tag}.csv"
    save_csv(validation_sweep, validation_sweep_path, index=False)
    selected_hook_name = str(selected["hook_name"])
    selected_alpha = float(selected["alpha"])
    selected_steering_vector = concept_steering_vector(
        vectors[selected_hook_name],
        normalize=config.normalize_concept_vector,
    )

    selected_path = output_dir / f"selected_site_{run_tag}.json"
    save_json(
        selected_path,
        {
            "selection_metric": "smallest_abs_alpha_within_fraction_of_best_signed_effect",
            "selection_effect_fraction": float(config.selection_effect_fraction),
            "selected": selected,
            "alpha_grid": list(config.alpha_grid),
            "normalize_concept_vector": bool(config.normalize_concept_vector),
            "split_counts": {split_name: int(len(split_df)) for split_name, split_df in splits.items()},
        },
    )

    test_summary, test_rows = evaluate_concept_steering(
        model=model,
        df=splits["test"],
        hook_name=selected_hook_name,
        concept_vector=selected_steering_vector,
        alpha=selected_alpha,
        animate_ids_tensor=prepared["animate_ids_tensor"],
        inanimate_ids_tensor=prepared["inanimate_ids_tensor"],
        batch_size=config.steering_batch_size,
        split_name="test",
        return_rows=True,
    )
    assert test_rows is not None
    test_rows_path = output_dir / f"test_steering_rows_{run_tag}.csv"
    save_csv(test_rows, test_rows_path, index=False)

    random_control_rows: list[dict[str, Any]] = []
    for repeat_index in tqdm(
        range(int(config.random_control_repeats)),
        desc="Random concept controls",
        unit="repeat",
    ):
        control_vector = random_control_vector(
            selected_steering_vector,
            seed=config.seed,
            repeat_index=repeat_index,
        )
        control_summary, _ = evaluate_concept_steering(
            model=model,
            df=splits["test"],
            hook_name=selected_hook_name,
            concept_vector=control_vector,
            alpha=selected_alpha,
            animate_ids_tensor=prepared["animate_ids_tensor"],
            inanimate_ids_tensor=prepared["inanimate_ids_tensor"],
            batch_size=config.steering_batch_size,
            split_name=f"random_control_{repeat_index}",
            return_rows=False,
        )
        random_control_rows.append(
            {
                "repeat": int(repeat_index),
                "hook_name": selected_hook_name,
                "alpha": selected_alpha,
                "control_vector_norm": float(control_vector.norm().item()),
                **control_summary,
            }
        )
    random_control_frame = pd.DataFrame(random_control_rows)
    random_control_path = output_dir / f"random_control_test_summary_{run_tag}.csv"
    save_csv(random_control_frame, random_control_path, index=False)
    random_control_summary = (
        {
            "repeat_count": int(len(random_control_frame)),
            "signed_effect_mean": float(random_control_frame["signed_effect_mean"].mean()),
            "signed_effect_std": (
                float(random_control_frame["signed_effect_mean"].std(ddof=0))
                if len(random_control_frame) > 1
                else 0.0
            ),
        }
        if not random_control_frame.empty
        else {"repeat_count": 0, "signed_effect_mean": 0.0, "signed_effect_std": 0.0}
    )

    summary_path = output_dir / f"concept_extraction_summary_{run_tag}.json"
    summary = {
        "config": asdict(config),
        "requested_model_name": prepared["requested_model_name"],
        "model_name": prepared["model_name"],
        "requested_dataset_filter_model_name": prepared["requested_dataset_filter_model_name"],
        "dataset_filter_model_name": prepared["dataset_filter_model_name"],
        "source_success_cache_path": prepared["source_success_cache_path"],
        "source_success_cache_status": prepared["source_success_cache_status"],
        "target_raw_accuracy": prepared["target_raw_accuracy"],
        "target_on_source_accuracy": prepared["target_on_source_accuracy"],
        "split_counts": {split_name: int(len(split_df)) for split_name, split_df in splits.items()},
        "selected": selected,
        "test_summary": test_summary,
        "random_control_summary": random_control_summary,
        "paths": {
            "output_dir": str(output_dir),
            "vectors": str(vectors_path),
            "site_summary": str(site_summary_path),
            "validation_sweep": str(validation_sweep_path),
            "selected_site": str(selected_path),
            "test_rows": str(test_rows_path),
            "random_control_test_summary": str(random_control_path),
            "splits": split_paths,
        },
    }
    summary["paths"]["summary"] = str(summary_path)
    save_json(summary_path, summary)
    return summary


def prepare_dataset_set_model_inputs(
    project_root: Path,
    model_name: str,
    dataset_set_name: str,
    metric_batch_size: int,
    seed: int,
    shared_filter_model_names: Sequence[str] = (),
    dataset_filter_path: Path | str | None = None,
    refresh_dataset_filter: bool = False,
    cache_dataset_filter: bool = True,
    max_filter_examples: int | None = None,
    target_filter_policy: str = "model_success",
    target_source: str | Path | None = None,
) -> dict[str, Any]:
    if dataset_set_name not in DATASET_SET_NAMES:
        raise ValueError(
            f"dataset_set_name must be one of {DATASET_SET_NAMES}, got {dataset_set_name!r}."
        )

    resolved_model_name = canonical_model_name(model_name)
    source_slug = target_source_slug(project_root, target_source)
    common_filter_model_names = unique_model_names(DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS)
    dataset_membership_models = (
        unique_model_names([*shared_filter_model_names, resolved_model_name])
        if dataset_set_name == SHARED_CORRECT
        else [resolved_model_name]
    )
    print(f"Loading target model {resolved_model_name} for {dataset_set_name}.")
    context = load_model_context(
        project_root,
        model_name,
        target_filter_model_names=common_filter_model_names,
        target_source=target_source,
    )
    raw_dataset_with_metadata = attach_pair_metadata(
        load_common_tokenized_pairs(project_root, common_filter_model_names),
        project_root,
    )
    raw_tokenization_diagnostics = token_alignment_diagnostics(
        raw_dataset_with_metadata,
        tokenizer=context["tokenizer"],
    )
    cached_scored_df = (
        load_metric_scored_model_dataset(project_root, resolved_model_name)
        if source_slug == DEFAULT_TARGET_SOURCE
        and target_filter_policy == "model_success"
        and not refresh_dataset_filter
        else None
    )
    if cached_scored_df is not None and max_filter_examples is None:
        target_raw_scored_df = cached_scored_df
    else:
        tokenization_source_df = load_tokenization_filtered_pairs_for_model(
            project_root,
            resolved_model_name,
        )
        target_raw_scored_df = compute_model_scored_dataset(
            project_root=project_root,
            model=context["model"],
            tokenizer=context["tokenizer"],
            animate_ids_tensor=context["animate_ids_tensor"],
            inanimate_ids_tensor=context["inanimate_ids_tensor"],
            batch_size=metric_batch_size,
            source_df=tokenization_source_df,
            seed=seed,
            max_examples=max_filter_examples,
        )
    cached_model_specific_df = (
        load_metric_filtered_model_success_dataset(project_root, resolved_model_name)
        if source_slug == DEFAULT_TARGET_SOURCE
        and target_filter_policy == "model_success"
        and not refresh_dataset_filter
        and max_filter_examples is None
        and (
            metric_scored_model_dataset_path(project_root, resolved_model_name).is_file()
            or find_metric_filtered_model_dataset_path(project_root, resolved_model_name) is not None
        )
        else None
    )
    model_specific_df = (
        cached_model_specific_df
        if cached_model_specific_df is not None
        else apply_target_filter_policy(target_raw_scored_df, target_filter_policy)
    )

    per_model_filtered_counts = {resolved_model_name: int(len(model_specific_df))}
    shared_prompt_pairs = prompt_pair_columns(model_specific_df)
    source_success_cache_path = None
    source_success_cache_status = None
    shared_candidate_df = None

    if dataset_set_name == SHARED_CORRECT:
        requested_membership_models = dataset_membership_models
        if len(requested_membership_models) < 2:
            raise ValueError(
                "shared_correct requires at least one non-target shared filter model."
            )
        dataset_membership_models = requested_membership_models
        prompt_pair_frames = [model_specific_df]

        for membership_model_name in requested_membership_models:
            if membership_model_name == resolved_model_name:
                continue
            filtered_df = load_policy_filtered_dataset(
                project_root=project_root,
                model_name=membership_model_name,
                batch_size=metric_batch_size,
                target_filter_policy=target_filter_policy,
                cache_path=dataset_filter_path if len(requested_membership_models) == 2 else None,
                refresh=refresh_dataset_filter,
                cache=cache_dataset_filter or dataset_filter_path is not None,
                max_examples=max_filter_examples,
                seed=seed,
                target_source=target_source,
            )
            prompt_pair_frames.append(filtered_df)
            per_model_filtered_counts[membership_model_name] = int(len(filtered_df))
            if source_success_cache_path is None:
                source_success_cache_path = filtered_df.attrs.get("model_success_cache_path")
                source_success_cache_status = filtered_df.attrs.get("model_success_cache_status")

        shared_prompt_pairs = intersect_prompt_pair_frames(prompt_pair_frames)
        shared_candidate_df = filter_df_to_prompt_pairs(target_raw_scored_df, shared_prompt_pairs)
        filtered_df = apply_target_filter_policy(shared_candidate_df, target_filter_policy)
    else:
        filtered_df = model_specific_df

    if dataset_set_name == MODEL_SPECIFIC_CORRECT:
        shared_candidate_df = filtered_df.copy()

    return {
        "dataset_set_name": dataset_set_name,
        "dataset_membership_models": dataset_membership_models,
        "per_model_filtered_counts": per_model_filtered_counts,
        "source_success_cache_path": source_success_cache_path,
        "source_success_cache_status": source_success_cache_status,
        "target_raw_scored_df": target_raw_scored_df,
        "model_specific_df": model_specific_df,
        "shared_candidate_df": shared_candidate_df,
        "filtered_df": filtered_df,
        "raw_tokenization_diagnostics": raw_tokenization_diagnostics,
        "target_raw_accuracy": task_accuracy_summary(target_raw_scored_df),
        "model_specific_accuracy": task_accuracy_summary(model_specific_df),
        "shared_candidate_accuracy": task_accuracy_summary(shared_candidate_df),
        "requested_model_name": model_name,
        "model_name": resolved_model_name,
        **context,
    }


def run_tokenization_safety_check(
    model_name: str = "gpt2",
    start: Path | str | None = None,
    target_source: str | Path | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    resolved_model_name = canonical_model_name(model_name)
    tokenizer = load_hf_tokenizer(model_name)
    animate_words, inanimate_words = load_animacy_targets(project_root, target_source=target_source)
    filtered_animate, filtered_inanimate, target_filter_summary, target_filter_path = (
        load_or_filter_targets_for_models(
            project_root,
            [resolved_model_name],
            target_tokenizer=tokenizer,
            target_source=target_source,
        )
    )
    raw_dataset_with_metadata = attach_pair_metadata(
        load_animacy_dataframe(project_root),
        project_root,
    )
    diagnostics = {
        "requested_model_name": model_name,
        "model_name": resolved_model_name,
        "model_note": model_note(resolved_model_name),
        "target_source": str(target_source or DEFAULT_TARGET_SOURCE),
        "target_source_path": str(resolve_target_source_path(project_root, target_source)),
        "raw_dataset_alignment": token_alignment_diagnostics(
            raw_dataset_with_metadata,
            tokenizer=tokenizer,
        ),
        "target_sets": build_target_tokenization_diagnostics(
            animate_words,
            inanimate_words,
            tokenizer,
        ),
        "filtered_target_sets": build_target_tokenization_diagnostics(
            filtered_animate,
            filtered_inanimate,
            tokenizer,
        ),
        "target_filter_summary": target_filter_summary,
        "target_filter_path": str(target_filter_path) if target_filter_path is not None else None,
    }
    return diagnostics


def sample_discovery_validation(
    df: pd.DataFrame,
    discovery_sample_size: int,
    seed: int,
    discovery_margin_threshold: float | None = DEFAULT_DISCOVERY_MARGIN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    discovery_pool = filter_discovery_margin_candidates(df, discovery_margin_threshold)
    if len(discovery_pool) < discovery_sample_size:
        raise ValueError(
            f"Requested {discovery_sample_size} discovery examples, but only "
            f"{len(discovery_pool)} examples satisfy the discovery margin threshold "
            f"{discovery_margin_threshold}."
        )

    discovery = discovery_pool.sample(n=discovery_sample_size, random_state=seed).copy()
    validation = df.drop(index=discovery.index).copy()
    if validation.empty:
        raise ValueError("Validation set is empty after removing discovery examples.")

    signature = prompt_pair_signature(discovery)
    return (
        discovery.reset_index(drop=True).copy(),
        validation.reset_index(drop=True).copy(),
        signature,
    )


def prepare_task_inputs(
    config: ComparisonConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    prepared = prepare_filtered_model_inputs(
        project_root=project_root,
        model_name=config.model_name,
        dataset_filter_model_name=config.dataset_filter_model_name,
        metric_batch_size=config.metric_batch_size,
        seed=config.seed,
        dataset_filter_path=config.dataset_filter_path,
        refresh_dataset_filter=config.refresh_dataset_filter,
        cache_dataset_filter=config.cache_dataset_filter,
        max_filter_examples=config.max_examples,
        target_filter_policy=config.target_filter_policy,
        target_source=config.target_source,
    )
    filtered_df = prepared["filtered_df"]
    discovery_df, validation_df = split_discovery_validation(
        filtered_df,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    return {
        "project_root": project_root,
        "model": prepared["model"],
        "tokenizer": prepared["tokenizer"],
        "animate_ids_tensor": prepared["animate_ids_tensor"],
        "inanimate_ids_tensor": prepared["inanimate_ids_tensor"],
        "source_success_df": prepared["source_success_df"],
        "target_raw_scored_df": prepared["target_raw_scored_df"],
        "target_scored_df": prepared["target_scored_df"],
        "raw_tokenization_diagnostics": prepared["raw_tokenization_diagnostics"],
        "target_tokenization_diagnostics": prepared["target_tokenization_diagnostics"],
        "target_raw_accuracy": prepared["target_raw_accuracy"],
        "target_on_source_accuracy": prepared["target_on_source_accuracy"],
        "source_success_cache_path": prepared["source_success_cache_path"],
        "source_success_cache_status": prepared["source_success_cache_status"],
        "filtered_df": filtered_df,
        "discovery_df": discovery_df,
        "validation_df": validation_df,
    }


def run_model_diagnostic(
    model_name: str = "gpt2",
    dataset_filter_model_name: str = "gpt2",
    filter_batch_size: int = 50,
    seed: int = 42,
    dataset_filter_path: Path | str | None = None,
    refresh_dataset_filter: bool = False,
    cache_dataset_filter: bool = True,
    max_filter_examples: int | None = None,
    target_filter_policy: str = "model_success",
    target_source: str | Path | None = None,
    output_day: str | None = None,
    output_dir: Path | str | None = None,
    save: bool = False,
    save_details: bool = False,
    save_debug_details: bool = False,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    prepared = prepare_filtered_model_inputs(
        project_root=project_root,
        model_name=model_name,
        dataset_filter_model_name=dataset_filter_model_name,
        metric_batch_size=filter_batch_size,
        seed=seed,
        dataset_filter_path=dataset_filter_path,
        refresh_dataset_filter=refresh_dataset_filter,
        cache_dataset_filter=cache_dataset_filter,
        max_filter_examples=max_filter_examples,
        target_filter_policy=target_filter_policy,
        target_source=target_source,
    )
    day = output_day or date_tag()
    paths: dict[str, str] = {"project_root": str(project_root)}

    artifact = {
        "experiment": "model_diagnostic",
        "config": {
            "model_name": model_name,
            "dataset_filter_model_name": dataset_filter_model_name,
            "filter_batch_size": filter_batch_size,
            "seed": seed,
            "dataset_filter_path": dataset_filter_path,
            "refresh_dataset_filter": refresh_dataset_filter,
            "cache_dataset_filter": cache_dataset_filter,
            "max_filter_examples": max_filter_examples,
            "target_filter_policy": target_filter_policy,
            "target_source": str(target_source or DEFAULT_TARGET_SOURCE),
            "target_source_path": str(resolve_target_source_path(project_root, target_source)),
            "output_day": day,
            "save": save,
            "save_details": save_details,
            "save_debug_details": save_debug_details,
        },
        "paths": paths,
        "dataset_summary": {
            "source_filter_model": prepared["dataset_filter_model_name"],
            "source_filter_model_requested": prepared["requested_dataset_filter_model_name"],
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "source_model_success_count": int(len(prepared["source_success_df"])),
            "source_success_cache_path": prepared["source_success_cache_path"],
            "source_success_cache_status": prepared["source_success_cache_status"],
            "target_raw_scored_count": int(len(prepared["target_raw_scored_df"])),
            "target_on_source_scored_count": int(len(prepared["target_scored_df"])),
            "target_filter_policy": target_filter_policy,
            "target_filtered_count": int(len(prepared["filtered_df"])),
            "target_raw_accuracy": prepared["target_raw_accuracy"],
            "target_on_source_accuracy": prepared["target_on_source_accuracy"],
            "target_filter_model_names": prepared["target_filter_model_names"],
            "target_filter_path": prepared["target_filter_path"],
            "target_source": prepared["target_source"],
            "target_source_path": prepared["target_source_path"],
        },
        "tokenization_diagnostics": {
            "raw_dataset_alignment": prepared["raw_tokenization_diagnostics"],
            "target_sets": prepared["target_tokenization_diagnostics"],
            "target_filter_summary": prepared["target_filter_summary"],
            "note": (
                "Target sets and sentence pairs are filtered before scoring. "
                "Sentence-pair filtering requires aligned clean/corrupt sequences "
                "and one-token patient/verb components in context."
            ),
        },
    }
    if save or output_dir is not None:
        if output_dir is not None:
            diagnostic_dir = ensure_generated_dir(
                Path(output_dir).resolve(),
                project_root,
                "model_diagnostic",
            )
        else:
            diagnostic_dir = ensure_generated_dir(
                model_diagnostic_dir(project_root, prepared["model_name"], day),
                project_root,
                "results",
                "model_diagnostic",
                safe_model_name(prepared["model_name"]),
                day,
            )

        tag = timestamp_tag()
        summary_path = diagnostic_dir / f"model_diagnostic_summary_{tag}.json"

        artifact["paths"].update(
            {
                "output_dir": str(diagnostic_dir),
                "summary": str(summary_path),
            }
        )
        if save_details or save_debug_details:
            target_filtered_path = diagnostic_dir / f"target_filtered_{tag}.csv"
            save_csv(prepared["filtered_df"], target_filtered_path, index=False)
            artifact["paths"].update(
                {
                    "target_filtered": str(target_filtered_path),
                }
            )
        if save_debug_details:
            target_raw_path = diagnostic_dir / f"target_raw_scored_{tag}.csv"
            target_on_source_path = diagnostic_dir / f"target_on_source_scored_{tag}.csv"
            save_csv(prepared["target_raw_scored_df"], target_raw_path, index=False)
            save_csv(prepared["target_scored_df"], target_on_source_path, index=False)
            artifact["paths"].update(
                {
                    "target_raw_scored": str(target_raw_path),
                    "target_on_source_scored": str(target_on_source_path),
                }
            )
        save_json(summary_path, artifact)
    return artifact


def run_model_diagnostic_for_dataset_set(
    *,
    model_name: str = "gpt2",
    dataset_set_name: str,
    shared_filter_model_names: Sequence[str] = (),
    filter_batch_size: int = 50,
    seed: int = 42,
    dataset_filter_path: Path | str | None = None,
    refresh_dataset_filter: bool = False,
    cache_dataset_filter: bool = True,
    max_filter_examples: int | None = None,
    target_filter_policy: str = "model_success",
    target_source: str | Path | None = None,
    output_day: str | None = None,
    save: bool = True,
    save_details: bool = False,
    save_debug_details: bool = False,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    prepared = prepare_dataset_set_model_inputs(
        project_root=project_root,
        model_name=model_name,
        dataset_set_name=dataset_set_name,
        metric_batch_size=filter_batch_size,
        seed=seed,
        shared_filter_model_names=shared_filter_model_names,
        dataset_filter_path=dataset_filter_path,
        refresh_dataset_filter=refresh_dataset_filter,
        cache_dataset_filter=cache_dataset_filter,
        max_filter_examples=max_filter_examples,
        target_filter_policy=target_filter_policy,
        target_source=target_source,
    )
    day = output_day or date_tag()
    diagnostic_dir = ensure_generated_dir(
        model_diagnostic_dataset_set_dir(project_root, prepared["model_name"], dataset_set_name, day),
        project_root,
        "results",
        "model_diagnostic",
        safe_model_name(prepared["model_name"]),
        dataset_set_name,
        day,
    )
    tag = timestamp_tag()
    summary_path = diagnostic_dir / f"model_diagnostic_summary_{tag}.json"

    artifact = {
        "experiment": "model_diagnostic",
        "config": {
            "model_name": model_name,
            "dataset_set_name": dataset_set_name,
            "shared_filter_model_names": list(shared_filter_model_names),
            "filter_batch_size": filter_batch_size,
            "seed": seed,
            "dataset_filter_path": dataset_filter_path,
            "refresh_dataset_filter": refresh_dataset_filter,
            "cache_dataset_filter": cache_dataset_filter,
            "max_filter_examples": max_filter_examples,
            "target_filter_policy": target_filter_policy,
            "output_day": day,
            "save": save,
            "save_details": save_details,
            "save_debug_details": save_debug_details,
        },
        "paths": {
            "project_root": str(project_root),
            "output_dir": str(diagnostic_dir),
            "summary": str(summary_path),
        },
        "dataset_summary": {
            "dataset_set_name": dataset_set_name,
            "dataset_membership_models": prepared["dataset_membership_models"],
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "target_filter_policy": target_filter_policy,
            "full_aligned_count": int(len(prepared["target_raw_scored_df"])),
            "model_specific_correct_count": int(len(prepared["model_specific_df"])),
            "shared_candidate_count": int(len(prepared["shared_candidate_df"])),
            "final_retained_count": int(len(prepared["filtered_df"])),
            "per_model_filtered_counts": prepared["per_model_filtered_counts"],
            "prompt_pair_signature": prompt_pair_signature(prepared["filtered_df"]),
            "source_success_cache_path": prepared["source_success_cache_path"],
            "source_success_cache_status": prepared["source_success_cache_status"],
            "target_raw_accuracy": prepared["target_raw_accuracy"],
            "model_specific_accuracy": prepared["model_specific_accuracy"],
            "shared_candidate_accuracy": prepared["shared_candidate_accuracy"],
            "target_filter_model_names": prepared["target_filter_model_names"],
            "target_filter_path": prepared["target_filter_path"],
        },
        "tokenization_diagnostics": {
            "raw_dataset_alignment": prepared["raw_tokenization_diagnostics"],
            "target_sets": prepared["target_tokenization_diagnostics"],
            "target_filter_summary": prepared["target_filter_summary"],
            "note": (
                "Target sets and sentence pairs are filtered before scoring. "
                "Sentence-pair filtering requires aligned clean/corrupt sequences "
                "and one-token patient/verb components in context."
            ),
        },
    }
    if save_details or save_debug_details:
        target_filtered_path = diagnostic_dir / f"target_filtered_{tag}.csv"
        save_csv(prepared["filtered_df"], target_filtered_path, index=False)
        artifact["paths"]["target_filtered"] = str(target_filtered_path)
    if save_debug_details:
        target_raw_path = diagnostic_dir / f"target_raw_scored_{tag}.csv"
        shared_candidate_path = diagnostic_dir / f"shared_candidate_scored_{tag}.csv"
        save_csv(prepared["target_raw_scored_df"], target_raw_path, index=False)
        save_csv(prepared["shared_candidate_df"], shared_candidate_path, index=False)
        artifact["paths"]["target_raw_scored"] = str(target_raw_path)
        artifact["paths"]["shared_candidate_scored"] = str(shared_candidate_path)
    if save:
        save_json(summary_path, artifact)
    return artifact


def make_eap_metrics(
    animate_ids_tensor: torch.Tensor,
    inanimate_ids_tensor: torch.Tensor,
) -> dict[str, Callable[..., torch.Tensor]]:
    return {
        "attribute": make_eap_normalized_recovery_metric(
            animate_ids_tensor,
            inanimate_ids_tensor,
        ),
        "faithfulness": make_eap_normalized_recovery_vector_metric(
            animate_ids_tensor,
            inanimate_ids_tensor,
        ),
        "accuracy": make_eap_accuracy_metric(
            animate_ids_tensor,
            inanimate_ids_tensor,
        ),
    }


def build_dynamic_eap_budget_grid(
    ranked_edge_count: int,
    *,
    budget_max_fraction: float,
    budget_floor: int,
    budget_tail_points: int,
    fixed_budget_prefix: Sequence[int] = DEFAULT_EAP_FIXED_BUDGET_PREFIX,
) -> list[int]:
    if ranked_edge_count <= 0:
        return []
    if budget_max_fraction <= 0:
        raise ValueError("budget_max_fraction must be positive.")
    if budget_floor <= 0:
        raise ValueError("budget_floor must be positive.")
    if budget_tail_points <= 0:
        raise ValueError("budget_tail_points must be positive.")

    requested_k_max = max(budget_floor, math.ceil(budget_max_fraction * ranked_edge_count))
    effective_k_max = min(ranked_edge_count, requested_k_max)

    grid = {
        int(budget)
        for budget in fixed_budget_prefix
        if 0 < int(budget) <= effective_k_max
    }
    if not grid:
        grid.add(effective_k_max)

    if effective_k_max > fixed_budget_prefix[-1]:
        tail_values = np.geomspace(
            fixed_budget_prefix[-1],
            effective_k_max,
            num=budget_tail_points,
        )
        grid.update(
            int(math.ceil(value))
            for value in tail_values
            if 0 < int(math.ceil(value)) <= effective_k_max
        )
        grid.add(effective_k_max)

    return sorted(grid)


def resolve_eap_budget_grid(
    ranked_edge_count: int,
    *,
    budgets: Sequence[int] | None,
    budget_max_fraction: float,
    budget_floor: int,
    budget_tail_points: int,
) -> list[int]:
    if budgets is not None:
        return sorted({int(budget) for budget in budgets if 0 < int(budget) <= ranked_edge_count})
    return build_dynamic_eap_budget_grid(
        ranked_edge_count,
        budget_max_fraction=budget_max_fraction,
        budget_floor=budget_floor,
        budget_tail_points=budget_tail_points,
    )


def run_eap_budget_sweep(
    model: HookedTransformer,
    scored_graph: Graph,
    ranked_edges: Sequence[dict[str, Any]],
    validation_loader: DataLoader,
    faithfulness_metric: Callable[..., torch.Tensor],
    accuracy_metric: Callable[..., torch.Tensor],
    budgets: Sequence[int],
    *,
    early_stop: bool = False,
    early_stop_threshold: float = DEFAULT_EAP_EARLY_STOP_THRESHOLD,
    early_stop_patience: int = DEFAULT_EAP_EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = DEFAULT_EAP_EARLY_STOP_MIN_DELTA,
    early_stop_start_budget: int = DEFAULT_EAP_EARLY_STOP_START_BUDGET,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from eap.evaluate import evaluate_graph

    ranked_edge_count = len(ranked_edges)
    budget_grid = [budget for budget in budgets if budget <= ranked_edge_count]
    if not budget_grid and ranked_edges:
        budget_grid = [ranked_edge_count]

    budget_rows: list[dict[str, Any]] = []
    completed_budgets: set[int] = set()
    resume_from_budget: int | None = None
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_frame = pd.read_csv(checkpoint_path)
        if not checkpoint_frame.empty and "collapsed_edge_budget" in checkpoint_frame.columns:
            checkpoint_frame = checkpoint_frame.copy()
            checkpoint_frame["collapsed_edge_budget"] = checkpoint_frame["collapsed_edge_budget"].astype(int)
            checkpoint_frame = checkpoint_frame.sort_values("collapsed_edge_budget").drop_duplicates(
                subset=["collapsed_edge_budget"],
                keep="last",
            )
            budget_rows = checkpoint_frame.to_dict("records")
            completed_budgets = {
                int(budget) for budget in checkpoint_frame["collapsed_edge_budget"].tolist()
            }
            if completed_budgets:
                resume_from_budget = max(completed_budgets)
    early_stop_metadata = {
        "enabled": bool(early_stop),
        "triggered": False,
        "threshold": float(early_stop_threshold),
        "patience": int(early_stop_patience),
        "min_delta": float(early_stop_min_delta),
        "start_budget": int(early_stop_start_budget),
        "reason": None,
        "last_evaluated_budget": resume_from_budget,
        "resume_from_budget": resume_from_budget,
    }
    if early_stop and budget_rows:
        for row in budget_rows:
            if float(row.get("faithfulness_mean", float("-inf"))) >= early_stop_threshold:
                early_stop_metadata["triggered"] = True
                early_stop_metadata["reason"] = (
                    "faithfulness_threshold_reached_from_checkpoint: "
                    f"budget {int(row['collapsed_edge_budget'])} already reached "
                    f"faithfulness {float(row['faithfulness_mean']):.4f} >= {early_stop_threshold:.2f}."
                )
                return pd.DataFrame(budget_rows), early_stop_metadata

    pending_budget_grid = [
        budget
        for budget in budget_grid
        if budget not in completed_budgets and (resume_from_budget is None or budget > resume_from_budget)
    ]

    for budget in tqdm(pending_budget_grid, desc="Greedy budget sweep"):
        candidate_graph = build_budget_circuit(scored_graph, ranked_edges, budget)
        faithfulness_values, accuracy_values = evaluate_graph(
            model,
            candidate_graph,
            validation_loader,
            [faithfulness_metric, accuracy_metric],
            quiet=True,
            intervention="patching",
            skip_clean=False,
        )
        faithfulness_mean = float(faithfulness_values.mean().item())
        accuracy_mean = float(accuracy_values.mean().item())
        budget_rows.append(
            {
                "collapsed_edge_budget": int(budget),
                "budget_fraction": float(budget / ranked_edge_count) if ranked_edge_count else 0.0,
                "expanded_edge_count": int(candidate_graph.count_included_edges()),
                "induced_node_count": int(candidate_graph.count_included_nodes() - 2),
                "faithfulness_mean": faithfulness_mean,
                "faithfulness_std": (
                    float(faithfulness_values.std(unbiased=False).item())
                    if len(faithfulness_values) > 1
                    else 0.0
                ),
                "accuracy_mean": accuracy_mean,
                "accuracy_std": (
                    float(accuracy_values.std(unbiased=False).item())
                    if len(accuracy_values) > 1
                    else 0.0
                ),
                "validation_examples": int(len(faithfulness_values)),
            }
        )
        if checkpoint_path is not None:
            save_csv(pd.DataFrame(budget_rows), checkpoint_path, index=False)
        early_stop_metadata["last_evaluated_budget"] = int(budget)

        if not early_stop:
            continue

        if faithfulness_mean >= early_stop_threshold:
            early_stop_metadata["triggered"] = True
            early_stop_metadata["reason"] = (
                "faithfulness_threshold_reached: "
                f"budget {budget} reached faithfulness {faithfulness_mean:.4f} "
                f">= {early_stop_threshold:.2f}."
            )
            break

    return pd.DataFrame(budget_rows), early_stop_metadata


def prepare_eap_experiment_inputs(
    config: EAPExperimentConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    prepared = prepare_filtered_model_inputs(
        project_root=project_root,
        model_name=config.model_name,
        dataset_filter_model_name=config.dataset_filter_model_name,
        metric_batch_size=config.filter_batch_size,
        seed=config.seed,
        dataset_filter_path=config.dataset_filter_path,
        refresh_dataset_filter=config.refresh_dataset_filter,
        cache_dataset_filter=config.cache_dataset_filter,
        max_filter_examples=config.max_filter_examples,
        target_filter_policy=config.target_filter_policy,
        target_source=config.target_source,
    )
    discovery_df, validation_df, sample_signature = sample_discovery_validation(
        prepared["filtered_df"],
        discovery_sample_size=config.discovery_sample_size,
        seed=config.seed,
        discovery_margin_threshold=config.discovery_margin_threshold,
    )

    return {
        "project_root": project_root,
        "discovery_df": discovery_df,
        "validation_df": validation_df,
        "sample_signature": sample_signature,
        **prepared,
    }


def prepare_dataset_set_eap_experiment_inputs(
    config: DualSetExperimentConfig,
    dataset_set_name: str,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    prepared = prepare_dataset_set_model_inputs(
        project_root=project_root,
        model_name=config.model_name,
        dataset_set_name=dataset_set_name,
        metric_batch_size=config.filter_batch_size,
        seed=config.seed,
        shared_filter_model_names=config.shared_filter_model_names,
        dataset_filter_path=config.dataset_filter_path,
        refresh_dataset_filter=config.refresh_dataset_filter,
        cache_dataset_filter=config.cache_dataset_filter,
        max_filter_examples=config.max_filter_examples,
        target_filter_policy=config.target_filter_policy,
        target_source=config.target_source,
    )
    discovery_df, validation_df, sample_signature = sample_discovery_validation(
        prepared["filtered_df"],
        discovery_sample_size=config.discovery_sample_size,
        seed=config.seed,
        discovery_margin_threshold=config.discovery_margin_threshold,
    )

    return {
        "project_root": project_root,
        "discovery_df": discovery_df,
        "validation_df": validation_df,
        "sample_signature": sample_signature,
        **prepared,
    }


def eap_artifact_base(
    config: EAPExperimentConfig,
    prepared: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "paths": {
            "project_root": str(prepared["project_root"]),
            "output_dir": str(output_dir),
        },
        "dataset_summary": {
            "source_filter_model": prepared["dataset_filter_model_name"],
            "source_filter_model_requested": prepared["requested_dataset_filter_model_name"],
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "source_model_success_count": int(len(prepared["source_success_df"])),
            "source_success_cache_path": prepared["source_success_cache_path"],
            "source_success_cache_status": prepared["source_success_cache_status"],
            "target_raw_scored_count": int(len(prepared["target_raw_scored_df"])),
            "target_on_source_scored_count": int(len(prepared["target_scored_df"])),
            "target_filter_policy": config.target_filter_policy,
            "target_filtered_count": int(len(prepared["filtered_df"])),
            "discovery_count": int(len(prepared["discovery_df"])),
            "validation_count": int(len(prepared["validation_df"])),
            "discovery_sample_signature": prepared["sample_signature"],
            "target_raw_accuracy": prepared["target_raw_accuracy"],
            "target_on_source_accuracy": prepared["target_on_source_accuracy"],
            "target_filter_model_names": prepared["target_filter_model_names"],
            "target_filter_path": prepared["target_filter_path"],
        },
        "tokenization_diagnostics": {
            "raw_dataset_alignment": prepared["raw_tokenization_diagnostics"],
            "target_sets": prepared["target_tokenization_diagnostics"],
            "target_filter_summary": prepared["target_filter_summary"],
            "note": (
                "Target sets and sentence pairs are filtered before scoring. "
                "Sentence-pair filtering requires aligned clean/corrupt sequences "
                "and one-token patient/verb components in context."
            ),
        },
    }


def dataset_set_eap_artifact_base(
    config: DualSetExperimentConfig,
    prepared: dict[str, Any],
    output_dir: Path,
    dataset_set_name: str,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "paths": {
            "project_root": str(prepared["project_root"]),
            "output_dir": str(output_dir),
        },
        "dataset_summary": {
            "dataset_set_name": dataset_set_name,
            "dataset_membership_models": prepared["dataset_membership_models"],
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "target_filter_policy": config.target_filter_policy,
            "full_aligned_count": int(len(prepared["target_raw_scored_df"])),
            "model_specific_correct_count": int(len(prepared["model_specific_df"])),
            "shared_candidate_count": int(len(prepared["shared_candidate_df"])),
            "final_retained_count": int(len(prepared["filtered_df"])),
            "per_model_filtered_counts": prepared["per_model_filtered_counts"],
            "discovery_count": int(len(prepared["discovery_df"])),
            "validation_count": int(len(prepared["validation_df"])),
            "discovery_sample_signature": prepared["sample_signature"],
            "prompt_pair_signature": prompt_pair_signature(prepared["filtered_df"]),
            "source_success_cache_path": prepared["source_success_cache_path"],
            "source_success_cache_status": prepared["source_success_cache_status"],
            "target_raw_accuracy": prepared["target_raw_accuracy"],
            "model_specific_accuracy": prepared["model_specific_accuracy"],
            "shared_candidate_accuracy": prepared["shared_candidate_accuracy"],
            "target_filter_model_names": prepared["target_filter_model_names"],
            "target_filter_path": prepared["target_filter_path"],
        },
        "tokenization_diagnostics": {
            "raw_dataset_alignment": prepared["raw_tokenization_diagnostics"],
            "target_sets": prepared["target_tokenization_diagnostics"],
            "target_filter_summary": prepared["target_filter_summary"],
            "note": (
                "Target sets and sentence pairs are filtered before scoring. "
                "Sentence-pair filtering requires aligned clean/corrupt sequences "
                "and one-token patient/verb components in context."
            ),
        },
    }


def run_full_model_eap_experiment(
    config: EAPExperimentConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    prepared = prepare_eap_experiment_inputs(config=config, start=start)
    project_root = prepared["project_root"]
    model = prepared["model"]
    output_day = config.output_day or date_tag()
    output_dir = eap_ig_full_model_dir(project_root, config.model_name, output_day)
    day = output_day
    edge_path = output_dir / f"full_model_edges_{day}.csv"
    node_path = output_dir / f"full_model_nodes_{day}.csv"
    budget_path = output_dir / f"full_model_budget_sweep_{day}.csv"
    budget_checkpoint_path = output_dir / f"full_model_budget_sweep_partial_{day}.csv"

    validation_loader = make_dataloader(
        prepared["validation_df"],
        batch_size=config.evaluation_batch_size,
        shuffle=False,
    )
    metrics = make_eap_metrics(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )

    saved_rankings = load_saved_ranked_edges(edge_path, node_path)
    if saved_rankings is None:
        discovery_loader = make_dataloader(
            prepared["discovery_df"],
            batch_size=config.attribution_batch_size,
            shuffle=False,
        )
        scored_graph = attribute_graph(
            model=model,
            graph=build_graph(model),
            dataloader=discovery_loader,
            metric=metrics["attribute"],
            ig_steps=config.ig_steps,
        )
        ranked_edges = collapsed_edge_groups(scored_graph)
        ranked_nodes = induced_node_ranking(ranked_edges)
        edge_frame = ranking_frame(ranked_edges)
        node_frame = ranking_frame(ranked_nodes)
        save_csv(edge_frame, edge_path, index=False)
        save_csv(node_frame, node_path, index=False)
    else:
        ranked_edges, edge_frame, node_frame = saved_rankings
        ranked_nodes = node_frame.to_dict("records")
        scored_graph = build_graph(model)
    resolved_budgets = resolve_eap_budget_grid(
        len(ranked_edges),
        budgets=config.budgets,
        budget_max_fraction=config.budget_max_fraction,
        budget_floor=config.budget_floor,
        budget_tail_points=config.budget_tail_points,
    )
    budget_frame, early_stop_summary = run_eap_budget_sweep(
        model=model,
        scored_graph=scored_graph,
        ranked_edges=ranked_edges,
        validation_loader=validation_loader,
        faithfulness_metric=metrics["faithfulness"],
        accuracy_metric=metrics["accuracy"],
        budgets=resolved_budgets,
        early_stop=config.budget_early_stop,
        early_stop_threshold=config.budget_early_stop_threshold,
        early_stop_patience=config.budget_early_stop_patience,
        early_stop_min_delta=config.budget_early_stop_min_delta,
        early_stop_start_budget=config.budget_early_stop_start_budget,
        checkpoint_path=budget_checkpoint_path,
    )

    save_csv(budget_frame, budget_path, index=False)
    visualization_paths = save_eap_visualizations(
        project_root=project_root,
        output_dir=output_dir,
        edge_frame=edge_frame,
        node_frame=node_frame,
        budget_frame=budget_frame,
        day=day,
    )

    artifact = eap_artifact_base(config, prepared, output_dir)
    artifact.update(
        {
            "experiment": "eap_ig_full_model",
            "paths": {
                **artifact["paths"],
                "edge_rankings": str(edge_path),
                "node_rankings": str(node_path),
                "budget_sweep": str(budget_path),
                "budget_sweep_partial": str(budget_checkpoint_path),
                "visualizations": visualization_paths,
            },
            "graph_summary": {
                "expanded_edge_count": int(scored_graph.real_edge_mask.sum().item()),
                "ranked_edge_count": len(ranked_edges),
                "ranked_node_count": len(ranked_nodes),
                "resolved_budget_grid": resolved_budgets,
                "budget_early_stop": early_stop_summary,
            },
        }
    )
    save_json(output_dir / f"full_model_summary_{day}.json", artifact)
    return artifact


def run_full_model_eap_experiment_for_dataset_set(
    config: DualSetExperimentConfig,
    dataset_set_name: str,
    start: Path | str | None = None,
) -> dict[str, Any]:
    prepared = prepare_dataset_set_eap_experiment_inputs(
        config=config,
        dataset_set_name=dataset_set_name,
        start=start,
    )
    project_root = prepared["project_root"]
    model = prepared["model"]
    output_day = config.output_day or date_tag()
    output_dir = eap_ig_dataset_set_full_model_dir(
        project_root,
        config.model_name,
        dataset_set_name,
        output_day,
    )
    day = output_day
    edge_path = output_dir / f"full_model_edges_{day}.csv"
    node_path = output_dir / f"full_model_nodes_{day}.csv"
    budget_path = output_dir / f"full_model_budget_sweep_{day}.csv"
    budget_checkpoint_path = output_dir / f"full_model_budget_sweep_partial_{day}.csv"

    validation_loader = make_dataloader(
        prepared["validation_df"],
        batch_size=config.evaluation_batch_size,
        shuffle=False,
    )
    metrics = make_eap_metrics(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )

    saved_rankings = load_saved_ranked_edges(edge_path, node_path)
    if saved_rankings is None:
        discovery_loader = make_dataloader(
            prepared["discovery_df"],
            batch_size=config.attribution_batch_size,
            shuffle=False,
        )
        scored_graph = attribute_graph(
            model=model,
            graph=build_graph(model),
            dataloader=discovery_loader,
            metric=metrics["attribute"],
            ig_steps=config.ig_steps,
        )
        ranked_edges = collapsed_edge_groups(scored_graph)
        ranked_nodes = induced_node_ranking(ranked_edges)
        edge_frame = ranking_frame(ranked_edges)
        node_frame = ranking_frame(ranked_nodes)
        save_csv(edge_frame, edge_path, index=False)
        save_csv(node_frame, node_path, index=False)
    else:
        ranked_edges, edge_frame, node_frame = saved_rankings
        ranked_nodes = node_frame.to_dict("records")
        scored_graph = build_graph(model)
    resolved_budgets = resolve_eap_budget_grid(
        len(ranked_edges),
        budgets=config.budgets,
        budget_max_fraction=config.budget_max_fraction,
        budget_floor=config.budget_floor,
        budget_tail_points=config.budget_tail_points,
    )
    budget_frame, early_stop_summary = run_eap_budget_sweep(
        model=model,
        scored_graph=scored_graph,
        ranked_edges=ranked_edges,
        validation_loader=validation_loader,
        faithfulness_metric=metrics["faithfulness"],
        accuracy_metric=metrics["accuracy"],
        budgets=resolved_budgets,
        early_stop=config.budget_early_stop,
        early_stop_threshold=config.budget_early_stop_threshold,
        early_stop_patience=config.budget_early_stop_patience,
        early_stop_min_delta=config.budget_early_stop_min_delta,
        early_stop_start_budget=config.budget_early_stop_start_budget,
        checkpoint_path=budget_checkpoint_path,
    )

    save_csv(budget_frame, budget_path, index=False)
    visualization_paths = save_eap_visualizations(
        project_root=project_root,
        output_dir=output_dir,
        edge_frame=edge_frame,
        node_frame=node_frame,
        budget_frame=budget_frame,
        day=day,
    )

    artifact = dataset_set_eap_artifact_base(config, prepared, output_dir, dataset_set_name)
    artifact.update(
        {
            "experiment": "eap_ig_full_model",
            "paths": {
                **artifact["paths"],
                "edge_rankings": str(edge_path),
                "node_rankings": str(node_path),
                "budget_sweep": str(budget_path),
                "budget_sweep_partial": str(budget_checkpoint_path),
                "visualizations": visualization_paths,
            },
            "graph_summary": {
                "expanded_edge_count": int(scored_graph.real_edge_mask.sum().item()),
                "ranked_edge_count": len(ranked_edges),
                "ranked_node_count": len(ranked_nodes),
                "resolved_budget_grid": resolved_budgets,
                "budget_early_stop": early_stop_summary,
            },
        }
    )
    save_json(output_dir / f"full_model_summary_{day}.json", artifact)
    return artifact


def _shadow_variant_removed_edges(
    *,
    variant_name: str,
    source_ranked_edges: Sequence[dict[str, Any]],
    config: EAPShadowRediscoveryConfig,
    first_threshold_row: dict[str, Any],
) -> list[dict[str, Any]]:
    top_k_match = re.fullmatch(r"remove_top_(\d+)_edges", variant_name)
    if top_k_match is not None:
        top_edge_count = int(top_k_match.group(1))
        return select_top_edge_groups(source_ranked_edges, top_edge_count)
    if variant_name == "remove_first_85pct_circuit":
        return select_top_edge_groups(
            source_ranked_edges,
            int(first_threshold_row["collapsed_edge_budget"]),
        )
    raise ValueError(f"Unknown shadow rediscovery variant: {variant_name}")


def run_shadow_rediscovery_variant(
    *,
    config: EAPShadowRediscoveryConfig,
    prepared: dict[str, Any],
    source_ranked_edges: Sequence[dict[str, Any]],
    source_budget_frame: pd.DataFrame,
    source_paths: dict[str, Path | None],
    source_provenance: dict[str, Any],
    first_threshold_row: dict[str, Any],
    variant_name: str,
    output_root: Path,
    day: str,
) -> dict[str, Any]:
    model = prepared["model"]
    variant_dir = ensure_dir(output_root / variant_name)
    edge_path = variant_dir / f"{variant_name}_edges_{day}.csv"
    node_path = variant_dir / f"{variant_name}_nodes_{day}.csv"
    removed_edge_path = variant_dir / f"{variant_name}_removed_edges_{day}.csv"
    budget_path = variant_dir / f"{variant_name}_budget_sweep_{day}.csv"
    budget_checkpoint_path = variant_dir / f"{variant_name}_budget_sweep_partial_{day}.csv"
    summary_path = variant_dir / f"{variant_name}_summary_{day}.json"

    removed_edges = _shadow_variant_removed_edges(
        variant_name=variant_name,
        source_ranked_edges=source_ranked_edges,
        config=config,
        first_threshold_row=first_threshold_row,
    )
    removed_underlying_names = underlying_edge_name_set(removed_edges)
    save_csv(ranking_frame(removed_edges), removed_edge_path, index=False)

    metrics = make_eap_metrics(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )
    validation_loader = make_dataloader(
        prepared["validation_df"],
        batch_size=config.evaluation_batch_size,
        shuffle=False,
    )

    saved_rankings = None
    if config.skip_existing:
        saved_rankings = load_saved_ranked_edges(edge_path, node_path)

    if saved_rankings is None:
        discovery_loader = make_dataloader(
            prepared["discovery_df"],
            batch_size=config.attribution_batch_size,
            shuffle=False,
        )
        reduced_graph, removed_underlying_edges = build_edge_removed_graph(model, removed_edges)
        scored_graph = attribute_graph(
            model=model,
            graph=reduced_graph,
            dataloader=discovery_loader,
            metric=metrics["attribute"],
            ig_steps=config.ig_steps,
        )
        ranked_edges = collapsed_edge_groups(scored_graph)
        ranked_nodes = induced_node_ranking(ranked_edges)
        edge_frame = ranking_frame(ranked_edges)
        node_frame = ranking_frame(ranked_nodes)
        save_csv(edge_frame, edge_path, index=False)
        save_csv(node_frame, node_path, index=False)
    else:
        ranked_edges, edge_frame, node_frame = saved_rankings
        ranked_nodes = node_frame.to_dict("records")
        scored_graph, removed_underlying_edges = build_edge_removed_graph(model, removed_edges)

    resolved_budgets = resolve_eap_budget_grid(
        len(ranked_edges),
        budgets=config.budgets,
        budget_max_fraction=config.budget_max_fraction,
        budget_floor=config.budget_floor,
        budget_tail_points=config.budget_tail_points,
    )
    budget_frame, early_stop_summary = run_eap_budget_sweep(
        model=model,
        scored_graph=scored_graph,
        ranked_edges=ranked_edges,
        validation_loader=validation_loader,
        faithfulness_metric=metrics["faithfulness"],
        accuracy_metric=metrics["accuracy"],
        budgets=resolved_budgets,
        early_stop=config.budget_early_stop,
        early_stop_threshold=config.budget_early_stop_threshold,
        early_stop_patience=config.budget_early_stop_patience,
        early_stop_min_delta=config.budget_early_stop_min_delta,
        early_stop_start_budget=config.budget_early_stop_start_budget,
        checkpoint_path=budget_checkpoint_path,
    )
    save_csv(budget_frame, budget_path, index=False)

    visualization_paths = save_eap_visualizations(
        project_root=prepared["project_root"],
        output_dir=variant_dir,
        edge_frame=edge_frame,
        node_frame=node_frame,
        budget_frame=budget_frame,
        day=day,
    )
    overlap = edge_overlap_summary(
        rediscovered_edges=ranked_edges,
        source_edges=source_ranked_edges,
        removed_edges=removed_edges,
    )
    artifact = {
        "experiment": "eap_ig_shadow_rediscovery",
        "variant": variant_name,
        "config": asdict(config),
        "paths": {
            "project_root": str(prepared["project_root"]),
            "output_dir": str(variant_dir),
            "edge_rankings": str(edge_path),
            "node_rankings": str(node_path),
            "removed_edge_rankings": str(removed_edge_path),
            "budget_sweep": str(budget_path),
            "budget_sweep_partial": str(budget_checkpoint_path),
            "summary": str(summary_path),
            "visualizations": visualization_paths,
            "source_dir": str(source_paths["source_dir"]) if source_paths.get("source_dir") is not None else None,
            "source_edge_rankings": str(source_paths["edge_path"]) if source_paths.get("edge_path") is not None else None,
            "source_budget_sweep": str(source_paths["budget_path"]) if source_paths.get("budget_path") is not None else None,
            "source_summary": str(source_paths["summary_path"]) if source_paths.get("summary_path") is not None else None,
        },
        "source": {
            "provenance": source_provenance,
            "faithfulness_threshold": float(config.source_faithfulness_threshold),
            "first_threshold_row": first_threshold_row,
            "source_budget_rows": int(len(source_budget_frame)),
        },
        "dataset_summary": {
            "dataset_set_name": config.dataset_set_name,
            "dataset_membership_models": prepared["dataset_membership_models"],
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "target_filter_policy": config.target_filter_policy,
            "discovery_count": int(len(prepared["discovery_df"])),
            "validation_count": int(len(prepared["validation_df"])),
            "discovery_sample_signature": prepared["sample_signature"],
            "prompt_pair_signature": prompt_pair_signature(prepared["filtered_df"]),
        },
        "graph_summary": {
            "source_ranked_edge_count": int(len(source_ranked_edges)),
            "removed_collapsed_edge_count": int(len(removed_edges)),
            "removed_underlying_edge_count": int(len(removed_underlying_edges)),
            "removed_underlying_edge_names_match": sorted(removed_underlying_names) == sorted(removed_underlying_edges),
            "remaining_expanded_edge_count": int(scored_graph.real_edge_mask.sum().item()),
            "ranked_edge_count": int(len(ranked_edges)),
            "ranked_node_count": int(len(ranked_nodes)),
            "resolved_budget_grid": resolved_budgets,
            "budget_early_stop": early_stop_summary,
            "overlap": overlap,
        },
    }
    save_json(summary_path, artifact)
    return artifact


def run_shadow_rediscovery_experiment(
    config: EAPShadowRediscoveryConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    if config.dataset_set_name not in DATASET_SET_NAMES:
        raise ValueError(
            f"dataset_set_name must be one of {DATASET_SET_NAMES}, got {config.dataset_set_name!r}."
        )

    prepared = prepare_dataset_set_eap_experiment_inputs(
        config=config,
        dataset_set_name=config.dataset_set_name,
        start=start,
    )
    project_root = prepared["project_root"]
    day = config.output_day or date_tag()
    output_root = eap_ig_dataset_set_shadow_rediscovery_dir(
        project_root,
        config.model_name,
        config.dataset_set_name,
        day,
    )

    source_paths = resolve_shadow_source_artifacts(
        project_root=project_root,
        model_name=config.model_name,
        dataset_set_name=config.dataset_set_name,
        main_experiment_path=config.main_experiment_path,
    )
    source_rankings = load_saved_ranked_edges(
        Path(source_paths["edge_path"]),
        Path(source_paths["node_path"]) if source_paths.get("node_path") is not None else None,
    )
    if source_rankings is None:
        raise ValueError(f"Could not load source edge ranking from {source_paths['edge_path']}")
    source_ranked_edges, source_edge_frame, source_node_frame = source_rankings
    del source_edge_frame, source_node_frame

    source_budget_frame = pd.read_csv(Path(source_paths["budget_path"]))
    first_threshold_row = first_budget_reaching_faithfulness(
        source_budget_frame,
        config.source_faithfulness_threshold,
    )
    source_provenance = validate_shadow_source_provenance(
        source_summary_path=Path(source_paths["summary_path"]) if source_paths.get("summary_path") is not None else None,
        prepared=prepared,
        config=config,
    )

    selected_variants = config.variants or ("top_k", "first_85pct")
    variants: list[str] = []
    for variant_key in selected_variants:
        if variant_key == "top_k":
            variants.append(f"remove_top_{int(config.top_edge_count)}_edges")
            continue
        if variant_key == "first_85pct":
            variants.append("remove_first_85pct_circuit")
            continue
        raise ValueError(f"Unknown shadow rediscovery variant selector: {variant_key!r}")
    artifacts = {
        variant_name: run_shadow_rediscovery_variant(
            config=config,
            prepared=prepared,
            source_ranked_edges=source_ranked_edges,
            source_budget_frame=source_budget_frame,
            source_paths=source_paths,
            source_provenance=source_provenance,
            first_threshold_row=first_threshold_row,
            variant_name=variant_name,
            output_root=output_root,
            day=day,
        )
        for variant_name in variants
    }

    summary = {
        "experiment": "eap_ig_shadow_rediscovery",
        "config": asdict(config),
        "paths": {
            "project_root": str(project_root),
            "output_dir": str(output_root),
            "source_dir": str(source_paths["source_dir"]) if source_paths.get("source_dir") is not None else None,
            "source_edge_rankings": str(source_paths["edge_path"]) if source_paths.get("edge_path") is not None else None,
            "source_budget_sweep": str(source_paths["budget_path"]) if source_paths.get("budget_path") is not None else None,
            "source_summary": str(source_paths["summary_path"]) if source_paths.get("summary_path") is not None else None,
        },
        "source": {
            "provenance": source_provenance,
            "faithfulness_threshold": float(config.source_faithfulness_threshold),
            "first_threshold_row": first_threshold_row,
            "source_ranked_edge_count": int(len(source_ranked_edges)),
        },
        "artifacts": artifacts,
    }
    summary_path = output_root / f"shadow_rediscovery_summary_{day}.json"
    summary["paths"]["summary"] = str(summary_path)
    save_json(summary_path, summary)
    return summary


def run_dual_set_eap_workflow(
    config: DualSetExperimentConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for dataset_set_name in config.dataset_set_names:
        if dataset_set_name not in DATASET_SET_NAMES:
            raise ValueError(
                f"dataset_set_name must be one of {DATASET_SET_NAMES}, got {dataset_set_name!r}."
            )
        run_bundle: dict[str, Any] = {}
        if config.run_diagnose:
            run_bundle["diagnose_model"] = run_model_diagnostic_for_dataset_set(
                model_name=config.model_name,
                dataset_set_name=dataset_set_name,
                shared_filter_model_names=config.shared_filter_model_names,
                filter_batch_size=config.filter_batch_size,
                seed=config.seed,
                dataset_filter_path=config.dataset_filter_path,
                refresh_dataset_filter=config.refresh_dataset_filter,
                cache_dataset_filter=config.cache_dataset_filter,
                max_filter_examples=config.max_filter_examples,
                target_filter_policy=config.target_filter_policy,
                output_day=config.output_day,
                save=True,
                start=start,
            )
        if config.run_eap:
            run_bundle["eap_full"] = run_full_model_eap_experiment_for_dataset_set(
                config=config,
                dataset_set_name=dataset_set_name,
                start=start,
            )
        artifacts[dataset_set_name] = run_bundle
    return {
        "experiment": "dual_set_eap",
        "config": asdict(config),
        "artifacts": artifacts,
    }


def select_thresholded_token_rows(
    data: dict[str, Any],
    token_label: str,
    occurrence: int = 0,
    quantile: float = 0.10,
) -> tuple[pd.DataFrame, float]:
    rows = heatmap_to_score_rows(data)
    token_position = get_saved_token_index(data, token_label, occurrence=occurrence)
    token_rows = rows[rows["token_position"] == token_position].copy()
    if token_rows.empty:
        raise ValueError(f"No rows found for token {token_label!r} at occurrence {occurrence}.")

    token_rows["abs_score"] = token_rows["score"].abs()
    token_threshold = float(token_rows["abs_score"].quantile(quantile))
    token_rows["token_threshold"] = token_threshold

    selected_rows = token_rows[token_rows["score"] > token_rows["token_threshold"]].copy()
    selected_rows = selected_rows.sort_values("score", ascending=False).reset_index(drop=True)
    return selected_rows, token_threshold


def has_legacy_component_checkpoints(results_dir: Path, day: str) -> bool:
    try:
        for spec in DISCOVERED_COMPONENT_SPECS:
            resolve_day_result_path(results_dir, day, spec["mlp_filename"])
            resolve_day_result_path(results_dir, day, spec["attn_filename"])
    except FileNotFoundError:
        return False
    return True


def legacy_component_checkpoint_days(
    results_dir: Path,
    model_name: str | None = None,
) -> set[str]:
    model_slugs = (
        {safe_model_name(canonical_model_name(model_name))}
        if model_name is not None
        else {"gpt2"}
    )
    days: set[str] = set()
    manual_root = results_dir / "manual_circuit_discovery"
    if manual_root.is_dir():
        for model_slug in model_slugs:
            model_dir = manual_root / model_slug
            if not model_dir.is_dir():
                continue
            for day_dir in model_dir.iterdir():
                if day_dir.is_dir():
                    days.add(day_dir.name)
    experiment_names = {
        spec["mlp_filename"].format(day="{day}").removesuffix(".pt").removesuffix("_{day}")
        for spec in DISCOVERED_COMPONENT_SPECS
    }
    for experiment_name in experiment_names:
        experiment_dir = results_dir / experiment_name
        if not experiment_dir.is_dir():
            continue
        for model_slug in model_slugs:
            model_dir = experiment_dir / model_slug
            if not model_dir.is_dir():
                continue
            for day_dir in model_dir.iterdir():
                if day_dir.is_dir():
                    days.add(day_dir.name)
    return {day for day in days if has_legacy_component_checkpoints(results_dir, day)}


def component_discovery_summary_candidates(
    results_dir: Path,
    day: str,
    model_name: str | None = None,
) -> list[Path]:
    filename = f"component_discovery_summary_{day}.json"
    candidates: list[Path] = []

    if model_name is not None:
        model_slug = safe_model_name(canonical_model_name(model_name))
        candidates.append(
            results_dir / "manual_circuit_discovery" / model_slug / day / "components" / filename
        )
        candidates.append(results_dir / "component_discovery" / model_slug / day / filename)
        legacy_day_dir = results_dir / day
        candidates.append(legacy_day_dir / model_slug / "component_discovery" / filename)

    candidates.append(results_dir / "manual_circuit_discovery" / day / "components" / filename)
    candidates.append(results_dir / "component_discovery" / day / filename)

    manual_dir = results_dir / "manual_circuit_discovery"
    if manual_dir.is_dir() and model_name is None:
        candidates.extend(sorted(manual_dir.glob(f"*/*/components/{filename}")))

    experiment_dir = results_dir / "component_discovery"
    if experiment_dir.is_dir() and model_name is None:
        candidates.extend(sorted(experiment_dir.glob(f"*/*/{filename}")))

    legacy_day_dir = results_dir / day
    candidates.append(legacy_day_dir / "component_discovery" / filename)
    if legacy_day_dir.is_dir() and model_name is None:
        candidates.extend(sorted(legacy_day_dir.glob(f"*/component_discovery/{filename}")))
        candidates.extend(sorted(legacy_day_dir.glob(f"**/{filename}")))

    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique.setdefault(str(candidate.resolve()), candidate)
    return list(unique.values())


def find_component_discovery_summary(
    results_dir: Path,
    day: str,
    model_name: str | None = None,
) -> Path | None:
    for candidate in component_discovery_summary_candidates(results_dir, day, model_name):
        if candidate.is_file():
            return candidate
    return None


def resolve_circuit_finder_day(
    results_dir: Path,
    requested_day: str | None = None,
    model_name: str | None = None,
) -> str:
    if requested_day is not None:
        return requested_day

    valid_days: set[str] = set()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    experiment_dir = results_dir / "component_discovery"
    if experiment_dir.is_dir():
        if model_name is not None:
            model_slug = safe_model_name(canonical_model_name(model_name))
            model_dir = experiment_dir / model_slug
            if model_dir.is_dir():
                for day_dir in model_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    if find_component_discovery_summary(
                        results_dir,
                        day_dir.name,
                        model_name=model_name,
                    ) is not None:
                        valid_days.add(day_dir.name)
        else:
            for model_dir in experiment_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for day_dir in model_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    if find_component_discovery_summary(
                        results_dir,
                        day_dir.name,
                        model_name=model_dir.name,
                    ) is not None:
                        valid_days.add(day_dir.name)

    valid_days.update(legacy_component_checkpoint_days(results_dir, model_name=model_name))

    for day_dir in sorted(results_dir.iterdir()):
        if not day_dir.is_dir() or not day_dir.name[:4].isdigit():
            continue
        has_new_summary = find_component_discovery_summary(
            results_dir,
            day_dir.name,
            model_name=model_name,
        ) is not None
        if has_new_summary or has_legacy_component_checkpoints(results_dir, day_dir.name):
            valid_days.add(day_dir.name)

    if not valid_days:
        raise FileNotFoundError(
            "Could not find any results day containing component-discovery outputs."
        )
    return sorted(valid_days)[-1]


def variant_threshold_value(variant: dict[str, Any]) -> float:
    try:
        return float(variant["threshold"])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def choose_component_discovery_variant(
    variants: Sequence[dict[str, Any]],
    requested_threshold: int | None,
) -> dict[str, Any]:
    if not variants:
        raise ValueError("Component-discovery summary does not contain any variants.")

    if requested_threshold is not None:
        matching = [
            variant
            for variant in variants
            if variant_threshold_value(variant) == float(requested_threshold)
        ]
        if not matching:
            available = sorted(
                {
                    int(variant_threshold_value(variant))
                    for variant in variants
                    if math.isfinite(variant_threshold_value(variant))
                }
            )
            raise ValueError(
                f"No component-discovery variant found for threshold {requested_threshold}. "
                f"Available thresholds: {available}"
            )
        return matching[0]

    with_nodes = [variant for variant in variants if variant.get("retained_nodes")]
    return max(with_nodes or list(variants), key=variant_threshold_value)


def infer_component_type_from_node(node: str) -> str:
    if re.fullmatch(r"m\d+", node):
        return "mlp"
    if re.fullmatch(r"a\d+\.h\d+", node):
        return "attn_head"
    return "component"


def component_node_rows_from_retained_nodes(
    retained_nodes: set[str],
    threshold: float,
    source_summary_path: Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "component_discovery",
                "component_type": infer_component_type_from_node(node),
                "node": node,
                "threshold": threshold,
                "checkpoint": str(source_summary_path),
            }
            for node in sorted(retained_nodes)
        ]
    )


def retained_module_rows_from_variant(
    variant: dict[str, Any],
    source_summary_path: Path,
) -> pd.DataFrame:
    rows_source = variant.get("retained_module_rows")
    if isinstance(rows_source, list):
        return pd.DataFrame(rows_source)
    if isinstance(rows_source, str) and rows_source:
        rows_path = Path(rows_source)
        if not rows_path.is_absolute():
            rows_path = source_summary_path.parent / rows_path
        if rows_path.is_file():
            return pd.read_csv(rows_path)
    return pd.DataFrame()


def expand_component_discovery_rows(
    retained_module_rows: pd.DataFrame,
    model: HookedTransformer,
    threshold: float,
    source_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if retained_module_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    expanded_rows: list[dict[str, Any]] = []
    for row in retained_module_rows.itertuples(index=False):
        row_dict = row._asdict()
        component = str(row_dict.get("component", ""))
        layer = int(row_dict["layer"]) if "layer" in row_dict else None
        if layer is None:
            continue

        base = {
            **row_dict,
            "source": "component_discovery",
            "threshold": threshold,
            "checkpoint": str(source_summary_path),
        }
        if component == "mlp":
            summary_rows.append(
                {
                    **base,
                    "component_type": "mlp",
                    "node": f"m{layer}",
                }
            )
            expanded_rows.append(summary_rows[-1].copy())
        elif component == "attn":
            summary_rows.append(
                {
                    **base,
                    "component_type": "attn_layer",
                    "node": f"a{layer}",
                }
            )
            for head in range(model.cfg.n_heads):
                expanded_rows.append(
                    {
                        **base,
                        "component_type": "attn_head",
                        "head": int(head),
                        "node": f"a{layer}.h{head}",
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(expanded_rows)


def build_component_selection_from_component_discovery(
    source_summary_path: Path,
    model: HookedTransformer,
    requested_threshold: int | None = None,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    artifact = json.loads(source_summary_path.read_text(encoding="utf-8"))
    variant = choose_component_discovery_variant(
        artifact.get("variants", []),
        requested_threshold=requested_threshold,
    )
    threshold = variant_threshold_value(variant)
    retained_nodes = {str(node) for node in variant.get("retained_nodes", [])}

    retained_module_rows = retained_module_rows_from_variant(variant, source_summary_path)
    summary_df, expanded_df = expand_component_discovery_rows(
        retained_module_rows,
        model=model,
        threshold=threshold,
        source_summary_path=source_summary_path,
    )

    if expanded_df.empty and retained_nodes:
        expanded_df = component_node_rows_from_retained_nodes(
            retained_nodes,
            threshold=threshold,
            source_summary_path=source_summary_path,
        )
    if summary_df.empty:
        summary_df = expanded_df.copy()

    expanded_nodes = set(expanded_df["node"].tolist()) if not expanded_df.empty else set()
    retained_nodes = retained_nodes or expanded_nodes
    return retained_nodes, summary_df.reset_index(drop=True), expanded_df.reset_index(drop=True)


def build_component_selection(
    results_dir: Path,
    day: str,
    model: HookedTransformer,
    quantile: float = 0.10,
    model_name: str | None = None,
    component_threshold: int | None = None,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    component_summary_path = find_component_discovery_summary(
        results_dir,
        day,
        model_name=model_name,
    )
    if component_summary_path is not None:
        return build_component_selection_from_component_discovery(
            component_summary_path,
            model=model,
            requested_threshold=component_threshold,
        )

    summary_frames: list[pd.DataFrame] = []
    expanded_frames: list[pd.DataFrame] = []

    for spec in DISCOVERED_COMPONENT_SPECS:
        mlp_path = resolve_day_result_path(results_dir, day, spec["mlp_filename"])
        mlp_rows, mlp_threshold = select_thresholded_token_rows(
            torch.load(mlp_path),
            spec["token_label"],
            occurrence=spec["occurrence"],
            quantile=quantile,
        )
        if not mlp_rows.empty:
            mlp_rows = mlp_rows.assign(
                source=spec["name"],
                component_type="mlp",
                checkpoint=str(mlp_path),
                threshold=mlp_threshold,
                node=mlp_rows["layer"].map(lambda layer: f"m{int(layer)}"),
            )
            summary_frames.append(mlp_rows.copy())
            expanded_frames.append(mlp_rows.copy())

        attn_path = resolve_day_result_path(results_dir, day, spec["attn_filename"])
        attn_rows, attn_threshold = select_thresholded_token_rows(
            torch.load(attn_path),
            spec["token_label"],
            occurrence=spec["occurrence"],
            quantile=quantile,
        )
        if not attn_rows.empty:
            attn_summary = attn_rows.assign(
                source=spec["name"],
                component_type="attn_layer",
                checkpoint=str(attn_path),
                threshold=attn_threshold,
            )
            summary_frames.append(attn_summary)

            expanded_head_rows: list[dict[str, Any]] = []
            for row in attn_rows.itertuples(index=False):
                for head in range(model.cfg.n_heads):
                    expanded_head_rows.append(
                        {
                            "source": spec["name"],
                            "component_type": "attn_head",
                            "token": row.token,
                            "token_position": int(row.token_position),
                            "token_position_from_end": int(row.token_position_from_end),
                            "layer": int(row.layer),
                            "head": int(head),
                            "score": float(row.score),
                            "abs_score": float(row.abs_score),
                            "token_threshold": float(row.token_threshold),
                            "threshold": float(attn_threshold),
                            "node": f"a{int(row.layer)}.h{head}",
                            "checkpoint": str(attn_path),
                        }
                    )
            expanded_frames.append(pd.DataFrame(expanded_head_rows))

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    expanded_df = pd.concat(expanded_frames, ignore_index=True) if expanded_frames else pd.DataFrame()

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["component_type", "token_position_from_end", "score"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    if not expanded_df.empty:
        expanded_df = expanded_df.sort_values(
            ["component_type", "token_position_from_end", "layer", "score"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

    retained_nodes = set(expanded_df["node"].tolist()) if not expanded_df.empty else set()
    return retained_nodes, summary_df, expanded_df


def run_selected_components_eap_experiment(
    config: EAPExperimentConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    prepared = prepare_eap_experiment_inputs(config=config, start=start)
    project_root = prepared["project_root"]
    model = prepared["model"]
    output_day = config.output_day or date_tag()
    circuit_finder_day = resolve_circuit_finder_day(
        project_root / "results",
        requested_day=config.circuit_finder_day,
        model_name=config.model_name,
    )
    output_dir = eap_ig_selected_components_dir(project_root, config.model_name, output_day)

    retained_nodes, component_summary_df, component_selection_df = build_component_selection(
        results_dir=project_root / "results",
        day=circuit_finder_day,
        model=model,
        quantile=config.importance_quantile,
        model_name=config.model_name,
        component_threshold=config.component_discovery_threshold,
    )
    if not retained_nodes:
        raise ValueError(
            "No components survived thresholding. Check circuit_finder_day or relax importance_quantile."
        )

    validation_loader = make_dataloader(
        prepared["validation_df"],
        batch_size=config.evaluation_batch_size,
        shuffle=False,
    )
    metrics = make_eap_metrics(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )

    full_graph = build_graph(model)
    reduced_graph, allowed_edge_names = build_reduced_graph(full_graph, retained_nodes)
    if not allowed_edge_names:
        raise ValueError("The reduced graph has no allowed edges for the retained nodes.")

    day = output_day
    component_summary_path = output_dir / f"selected_component_layers_{circuit_finder_day}.csv"
    component_nodes_path = output_dir / f"selected_component_nodes_{circuit_finder_day}.csv"
    edge_path = output_dir / f"selected_components_edges_{day}.csv"
    node_path = output_dir / f"selected_components_nodes_{day}.csv"
    budget_path = output_dir / f"selected_components_budget_sweep_{day}.csv"
    budget_checkpoint_path = output_dir / f"selected_components_budget_sweep_partial_{day}.csv"
    save_csv(component_summary_df, component_summary_path, index=False)
    save_csv(component_selection_df, component_nodes_path, index=False)
    saved_rankings = load_saved_ranked_edges(edge_path, node_path)
    if saved_rankings is None:
        discovery_loader = make_dataloader(
            prepared["discovery_df"],
            batch_size=config.attribution_batch_size,
            shuffle=False,
        )
        scored_graph = attribute_graph(
            model=model,
            graph=reduced_graph,
            dataloader=discovery_loader,
            metric=metrics["attribute"],
            ig_steps=config.ig_steps,
        )
        ranked_edges = collapsed_edge_groups(scored_graph)
        ranked_nodes = induced_node_ranking(ranked_edges)
        edge_frame = ranking_frame(ranked_edges)
        node_frame = ranking_frame(ranked_nodes)
        save_csv(edge_frame, edge_path, index=False)
        save_csv(node_frame, node_path, index=False)
    else:
        ranked_edges, edge_frame, node_frame = saved_rankings
        ranked_nodes = node_frame.to_dict("records")
        scored_graph = reduced_graph
    resolved_budgets = resolve_eap_budget_grid(
        len(ranked_edges),
        budgets=config.budgets,
        budget_max_fraction=config.budget_max_fraction,
        budget_floor=config.budget_floor,
        budget_tail_points=config.budget_tail_points,
    )
    budget_frame, early_stop_summary = run_eap_budget_sweep(
        model=model,
        scored_graph=scored_graph,
        ranked_edges=ranked_edges,
        validation_loader=validation_loader,
        faithfulness_metric=metrics["faithfulness"],
        accuracy_metric=metrics["accuracy"],
        budgets=resolved_budgets,
        early_stop=config.budget_early_stop,
        early_stop_threshold=config.budget_early_stop_threshold,
        early_stop_patience=config.budget_early_stop_patience,
        early_stop_min_delta=config.budget_early_stop_min_delta,
        early_stop_start_budget=config.budget_early_stop_start_budget,
        checkpoint_path=budget_checkpoint_path,
    )

    save_csv(budget_frame, budget_path, index=False)
    visualization_paths = save_eap_visualizations(
        project_root=project_root,
        output_dir=output_dir,
        edge_frame=edge_frame,
        node_frame=node_frame,
        budget_frame=budget_frame,
        day=day,
    )

    artifact = eap_artifact_base(config, prepared, output_dir)
    artifact.update(
        {
            "experiment": "eap_ig_selected_components",
            "circuit_finder_day": circuit_finder_day,
            "importance_quantile": config.importance_quantile,
            "component_discovery_threshold": config.component_discovery_threshold,
            "paths": {
                **artifact["paths"],
                "component_summary": str(component_summary_path),
                "component_nodes": str(component_nodes_path),
                "edge_rankings": str(edge_path),
                "node_rankings": str(node_path),
                "budget_sweep": str(budget_path),
                "budget_sweep_partial": str(budget_checkpoint_path),
                "visualizations": visualization_paths,
            },
            "graph_summary": {
                "retained_node_count": len(retained_nodes),
                "allowed_edge_count": len(allowed_edge_names),
                "expanded_edge_count": int(scored_graph.real_edge_mask.sum().item()),
                "ranked_edge_count": len(ranked_edges),
                "ranked_node_count": len(ranked_nodes),
                "resolved_budget_grid": resolved_budgets,
                "budget_early_stop": early_stop_summary,
            },
        }
    )
    save_json(output_dir / f"selected_components_summary_{day}.json", artifact)
    return artifact


def prepare_component_discovery_inputs(
    config: ComponentDiscoveryConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(start)
    prepared = prepare_filtered_model_inputs(
        project_root=project_root,
        model_name=config.model_name,
        dataset_filter_model_name=config.dataset_filter_model_name,
        metric_batch_size=config.filter_batch_size,
        seed=config.seed,
        dataset_filter_path=config.dataset_filter_path,
        refresh_dataset_filter=config.refresh_dataset_filter,
        cache_dataset_filter=config.cache_dataset_filter,
        max_filter_examples=config.max_filter_examples,
        target_filter_policy=config.target_filter_policy,
    )
    discovery_df, validation_df, sample_signature = sample_discovery_validation(
        prepared["filtered_df"],
        discovery_sample_size=config.discovery_sample_size,
        seed=config.seed,
        discovery_margin_threshold=config.discovery_margin_threshold,
    )
    return {
        "project_root": project_root,
        "discovery_df": discovery_df,
        "validation_df": validation_df,
        "sample_signature": sample_signature,
        **prepared,
    }


def run_component_discovery_experiment(
    config: ComponentDiscoveryConfig,
    start: Path | str | None = None,
) -> dict[str, Any]:
    from transformer_lens import patching as tl_patching

    prepared = prepare_component_discovery_inputs(config=config, start=start)
    project_root = prepared["project_root"]
    model = prepared["model"]
    tokenizer = prepared["tokenizer"]
    output_day = config.output_day or date_tag()
    output_dir = manual_circuit_discovery_dir(project_root, config.model_name, output_day)
    checkpoints_dir = manual_circuit_checkpoints_dir(project_root, config.model_name, output_day)
    components_dir = manual_circuit_components_dir(project_root, config.model_name, output_day)

    patch_metric_factory = make_avg_logit_difference_recovery_metric(
        prepared["animate_ids_tensor"],
        prepared["inanimate_ids_tensor"],
    )
    discovery_df = prepared["discovery_df"]

    residual_path = run_and_save_experiment(
        project_root=project_root,
        model=model,
        experiment_name="Residual_Stream_Patching",
        df=discovery_df,
        tokenizer=tokenizer,
        patching_func=tl_patching.get_act_patch_resid_pre,
        metric_factory=patch_metric_factory,
        filter_str="hook_resid_pre",
        batch_size=config.patch_batch_size,
        safety_checks=True,
        output_dir=checkpoints_dir,
    )
    residual_rows = heatmap_to_score_rows(torch.load(residual_path))
    positive_residual_rows = residual_rows[residual_rows["score"] > 0].copy()

    variants: list[dict[str, Any]] = []
    for threshold in config.thresholds:
        retained_residual_rows = select_positive_rows_at_percentile(
            positive_residual_rows,
            threshold,
        )
        retained_sites = patch_rows_to_sites(retained_residual_rows)

        if retained_sites:
            mlp_patch_path = run_and_save_experiment(
                project_root=project_root,
                model=model,
                experiment_name=f"Hybrid_MLP_Patching_t{threshold}",
                df=discovery_df,
                tokenizer=tokenizer,
                patching_func=make_multi_site_patching_func(retained_sites, component="mlp_out"),
                metric_factory=patch_metric_factory,
                filter_str="hook_mlp_out",
                batch_size=config.patch_batch_size,
                output_dir=checkpoints_dir,
            )
            attn_patch_path = run_and_save_experiment(
                project_root=project_root,
                model=model,
                experiment_name=f"Hybrid_Attn_Patching_t{threshold}",
                df=discovery_df,
                tokenizer=tokenizer,
                patching_func=make_multi_site_patching_func(retained_sites, component="attn_out"),
                metric_factory=patch_metric_factory,
                filter_str="hook_attn_out",
                batch_size=config.patch_batch_size,
                output_dir=checkpoints_dir,
            )
            mlp_rows = heatmap_to_score_rows(torch.load(mlp_patch_path)).assign(component="mlp")
            attn_rows = heatmap_to_score_rows(torch.load(attn_patch_path)).assign(component="attn")
            module_rows = pd.concat([mlp_rows, attn_rows], ignore_index=True)
            retained_module_rows = select_positive_rows_at_percentile(
                module_rows[module_rows["score"] > 0].copy(),
                threshold,
            )
        else:
            mlp_patch_path = None
            attn_patch_path = None
            retained_module_rows = pd.DataFrame(
                columns=[
                    "layer",
                    "token_position",
                    "token_position_from_end",
                    "token",
                    "score",
                    "component",
                ]
            )

        retained_mlp_layers = {
            int(layer)
            for layer in retained_module_rows.loc[
                retained_module_rows["component"] == "mlp", "layer"
            ].tolist()
        }
        retained_attn_layers = {
            int(layer)
            for layer in retained_module_rows.loc[
                retained_module_rows["component"] == "attn", "layer"
            ].tolist()
        }
        retained_nodes = {f"m{layer}" for layer in retained_mlp_layers}
        for layer in retained_attn_layers:
            retained_nodes.update(f"a{layer}.h{head}" for head in range(model.cfg.n_heads))

        retained_path = components_dir / f"component_discovery_threshold_{threshold}_{output_day}.csv"
        save_csv(retained_module_rows, retained_path, index=False)
        variants.append(
            {
                "threshold": threshold,
                "residual_site_count": int(len(retained_residual_rows)),
                "retained_module_count": int(len(retained_module_rows)),
                "retained_nodes": sorted(retained_nodes),
                "retained_mlp_layers": sorted(retained_mlp_layers),
                "retained_attention_block_layers": sorted(retained_attn_layers),
                "residual_patch_file": str(residual_path),
                "mlp_patch_file": str(mlp_patch_path) if mlp_patch_path else None,
                "attn_patch_file": str(attn_patch_path) if attn_patch_path else None,
                "retained_module_rows": str(retained_path),
            }
        )

    artifact = {
        "config": asdict(config),
        "experiment": "component_discovery",
        "paths": {
            "project_root": str(project_root),
            "output_dir": str(output_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "components_dir": str(components_dir),
            "residual_patch_file": str(residual_path),
        },
        "dataset_summary": {
            "source_filter_model": prepared["dataset_filter_model_name"],
            "source_filter_model_requested": prepared["requested_dataset_filter_model_name"],
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "source_model_success_count": int(len(prepared["source_success_df"])),
            "source_success_cache_path": prepared["source_success_cache_path"],
            "source_success_cache_status": prepared["source_success_cache_status"],
            "target_raw_scored_count": int(len(prepared["target_raw_scored_df"])),
            "target_on_source_scored_count": int(len(prepared["target_scored_df"])),
            "target_filter_policy": config.target_filter_policy,
            "target_filtered_count": int(len(prepared["filtered_df"])),
            "discovery_count": int(len(prepared["discovery_df"])),
            "discovery_sample_signature": prepared["sample_signature"],
            "target_raw_accuracy": prepared["target_raw_accuracy"],
            "target_on_source_accuracy": prepared["target_on_source_accuracy"],
        },
        "tokenization_diagnostics": {
            "raw_dataset_alignment": prepared["raw_tokenization_diagnostics"],
            "target_sets": prepared["target_tokenization_diagnostics"],
            "note": (
                "Alignment failures are counted for diagnostics only. "
                "They are not used as an additional filter here."
            ),
        },
        "variants": variants,
    }
    save_json(components_dir / f"component_discovery_summary_{output_day}.json", artifact)
    return artifact


def run_comparison(config: ComparisonConfig, start: Path | str | None = None) -> dict[str, Any]:
    prepared = prepare_task_inputs(config=config, start=start)
    project_root = prepared["project_root"]
    model = prepared["model"]
    tokenizer = prepared["tokenizer"]
    animate_ids_tensor = prepared["animate_ids_tensor"]
    inanimate_ids_tensor = prepared["inanimate_ids_tensor"]
    discovery_df = prepared["discovery_df"]
    validation_df = prepared["validation_df"]

    output_dir = experiment_output_dir(
        project_root,
        config.output_stem,
        config.model_name,
    )
    eap_metric = make_eap_normalized_recovery_metric(
        animate_ids_tensor,
        inanimate_ids_tensor,
    )

    pipeline_a = run_pipeline_a(
        project_root=project_root,
        output_dir=output_dir,
        model=model,
        discovery_df=discovery_df,
        validation_df=validation_df,
        metric=eap_metric,
        config=config,
    )
    pipeline_b = run_pipeline_b(
        project_root=project_root,
        output_dir=output_dir,
        model=model,
        tokenizer=tokenizer,
        discovery_df=discovery_df,
        validation_df=validation_df,
        animate_ids_tensor=animate_ids_tensor,
        inanimate_ids_tensor=inanimate_ids_tensor,
        config=config,
    )

    summaries: list[VariantSelectionSummary] = []
    pipeline_a_summary = variant_selection_summary(
        variant_id=pipeline_a["variant_id"],
        pipeline=pipeline_a["pipeline"],
        threshold=pipeline_a["threshold"],
        budget_results=[BudgetEvaluation(**result) for result in pipeline_a["budget_results"]],
    )
    if pipeline_a_summary is not None:
        summaries.append(pipeline_a_summary)

    for variant in pipeline_b["variants"]:
        summary = variant_selection_summary(
            variant_id=variant["variant_id"],
            pipeline=variant["pipeline"],
            threshold=variant["threshold"],
            budget_results=[BudgetEvaluation(**result) for result in variant["budget_results"]],
        )
        if summary is not None:
            summaries.append(summary)

    chosen_variant = choose_final_variant(
        summaries=summaries,
        tolerance=config.selection_tolerance,
    )

    hybrid_collapsed_sizes = [
        variant["reduced_graph_collapsed_edge_count"]
        for variant in pipeline_b["variants"]
    ]
    hybrid_expanded_sizes = [
        variant["reduced_graph_expanded_edge_count"]
        for variant in pipeline_b["variants"]
    ]

    comparison_artifact = {
        "config": asdict(config),
        "paths": {
            "project_root": str(project_root),
            "output_dir": str(output_dir),
        },
        "dataset_summary": {
            "source_filter_model": prepared["dataset_filter_model_name"],
            "source_filter_model_requested": prepared["requested_dataset_filter_model_name"],
            "target_model": prepared["model_name"],
            "target_model_requested": prepared["requested_model_name"],
            "source_model_success_count": int(len(prepared["source_success_df"])),
            "source_success_cache_path": prepared["source_success_cache_path"],
            "source_success_cache_status": prepared["source_success_cache_status"],
            "target_raw_scored_count": int(len(prepared["target_raw_scored_df"])),
            "target_on_source_scored_count": int(len(prepared["target_scored_df"])),
            "target_filter_policy": config.target_filter_policy,
            "target_filtered_count": int(len(prepared["filtered_df"])),
            "discovery_count": int(len(discovery_df)),
            "validation_count": int(len(validation_df)),
            "target_raw_accuracy": prepared["target_raw_accuracy"],
            "target_on_source_accuracy": prepared["target_on_source_accuracy"],
        },
        "tokenization_diagnostics": {
            "raw_dataset_alignment": prepared["raw_tokenization_diagnostics"],
            "target_sets": prepared["target_tokenization_diagnostics"],
            "note": (
                "Alignment failures are counted for diagnostics only. "
                "They are not used as an additional filter here."
            ),
        },
        "selection_rule": (
            "highest mean held-out faithfulness across supported matched budgets, "
            "then smaller mean induced node count. Stability disabled because only seed 42 was requested."
        ),
        "stability_summary": {
            "enabled": False,
            "reason": "single_seed_configuration",
            "seed": config.seed,
        },
        "sanity_checks": {
            "same_source_model_success_dataset": True,
            "shared_split_seed": config.seed,
            "pipeline_b_collapsed_graph_nonincreasing_across_thresholds": is_nonincreasing(
                hybrid_collapsed_sizes
            ),
            "pipeline_b_expanded_graph_nonincreasing_across_thresholds": is_nonincreasing(
                hybrid_expanded_sizes
            ),
        },
        "pipeline_a": pipeline_a,
        "pipeline_b": pipeline_b,
        "selection_summaries": [asdict(summary) for summary in summaries],
        "chosen_final_pipeline": chosen_variant,
    }
    save_json(output_dir / "comparison_summary.json", comparison_artifact)
    return comparison_artifact

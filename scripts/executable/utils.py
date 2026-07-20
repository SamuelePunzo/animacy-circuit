from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_ALIASES = {
    "gpt2": "gpt2",
    "gpt-2": "gpt2",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama 3.2 3b": "meta-llama/Llama-3.2-3B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama3.2 3b": "meta-llama/Llama-3.2-3B",
    "llama 3.2 3b base": "meta-llama/Llama-3.2-3B",
    "llama-3.2-3b-base": "meta-llama/Llama-3.2-3B",
    "meta-llama/llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "qwen-3-4b": "Qwen/Qwen3-4B",
    "qwen 3 4b": "Qwen/Qwen3-4B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen/qwen3-4b": "Qwen/Qwen3-4B",
    "gemma-3-4b-pt": "google/gemma-3-4b-pt",
    "gemma 3 4b pt": "google/gemma-3-4b-pt",
    "gemma3-4b-pt": "google/gemma-3-4b-pt",
    "gemma3 4b pt": "google/gemma-3-4b-pt",
    "gemma-3-4b-base": "google/gemma-3-4b-pt",
    "gemma 3 4b base": "google/gemma-3-4b-pt",
    "gemma3-4b-base": "google/gemma-3-4b-pt",
    "gemma3 4b base": "google/gemma-3-4b-pt",
    "google/gemma-3-4b-pt": "google/gemma-3-4b-pt",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma 2 2b": "google/gemma-2-2b",
    "google/gemma-2-2b": "google/gemma-2-2b",
}

MODEL_NOTES = {
    "Qwen/Qwen3-4B": (
        "Qwen3-4B is the active Qwen 4B HookedTransformer target in this environment."
    ),
    "google/gemma-3-4b-pt": (
        "Gemma 3 4B PT is the text base checkpoint to use for TransformerLens-based "
        "circuit discovery and EAP, not the instruction-tuned Gemma 3 variant."
    ),
    "meta-llama/Llama-3.2-3B": (
        "Llama 3.2 3B may require Hugging Face access approval and a logged-in token."
    ),
}


def normalize_model_alias_key(model_name: str) -> str:
    return re.sub(r"[\s_]+", "-", model_name.strip().casefold())


def canonical_model_name(model_name: str) -> str:
    stripped = model_name.strip()
    return MODEL_ALIASES.get(normalize_model_alias_key(stripped), stripped)


def model_note(model_name: str) -> str | None:
    return MODEL_NOTES.get(canonical_model_name(model_name))


def resolve_animacy_circuit_root(start: Path | str | None = None) -> Path:
    """Find the `animacy-circuit` project root from a nearby path or cwd."""
    anchor = Path(start) if start is not None else Path.cwd()
    anchor = anchor.resolve()
    if anchor.is_file():
        anchor = anchor.parent

    for base in (anchor, *anchor.parents):
        for candidate in (base, base / "animacy-circuit"):
            if (
                candidate.is_dir()
                and (candidate / "dataset").is_dir()
                and (candidate / "results").is_dir()
                and (candidate / "scripts").is_dir()
            ):
                return candidate

    raise FileNotFoundError(
        "Could not locate the animacy-circuit root. "
        "Expected a directory containing dataset/, results/, and scripts/."
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def temporary_sibling_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    tmp_path = temporary_sibling_path(path)
    tmp_path.write_text(text, encoding=encoding)
    tmp_path.replace(path)


def save_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = temporary_sibling_path(path)
    frame.to_csv(tmp_path, **kwargs)
    tmp_path.replace(path)


def save_torch(payload: Any, path: Path) -> None:
    import torch

    ensure_dir(path.parent)
    tmp_path = temporary_sibling_path(path)
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def timestamp_tag() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def date_tag() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def safe_model_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return slug or "model"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if converted is not value:
            return to_jsonable(converted)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except ValueError:
            converted = value
        if converted is not value:
            return to_jsonable(converted)

    return value


def json_default(value: Any) -> Any:
    converted = to_jsonable(value)
    if converted is not value:
        return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(to_jsonable(payload), indent=2, sort_keys=False, default=json_default),
        encoding="utf-8",
    )


# EAP-IG circuit inspection helpers
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
ATTN_RE = re.compile(r"^a(\d+)[._]h(\d+)$")
MLP_RE = re.compile(r"^m(\d+)$")


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "results").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root containing both ./results and ./scripts")


def normalize_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def matches_model_query(query: str, aliases: list[str]) -> bool:
    normalized_query = normalize_name(query)
    if not normalized_query:
        return False
    for alias in aliases:
        normalized_alias = normalize_name(alias)
        if not normalized_alias:
            continue
        if normalized_query in normalized_alias or normalized_alias in normalized_query:
            return True
    return False


def safe_read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for value in values:
        text = str(value or "").strip()
        if text:
            seen[text] = None
    return list(seen.keys())


def _display_model_from_summary(summary: dict, fallback: str) -> str:
    dataset_summary = summary.get("dataset_summary", {})
    config = summary.get("config", {})
    return (
        dataset_summary.get("target_model_requested")
        or dataset_summary.get("target_model")
        or config.get("model_name")
        or fallback
    )


def _model_aliases(summary: dict, fallback: str) -> list[str]:
    dataset_summary = summary.get("dataset_summary", {})
    config = summary.get("config", {})
    return _ordered_unique(
        [
            fallback,
            _display_model_from_summary(summary, fallback),
            dataset_summary.get("target_model"),
            dataset_summary.get("target_model_requested"),
            config.get("model_name"),
        ]
    )


def discover_model_diagnostics(project_root: Path) -> pd.DataFrame:
    rows = []
    base = project_root / "results" / "model_diagnostic"
    for summary_path in sorted(base.rglob("model_diagnostic_summary_*.json")):
        day = summary_path.parent.name
        model_dir_name = summary_path.parent.parent.name
        summary = safe_read_json(summary_path)
        rows.append(
            {
                "artifact_type": "model_diagnostic",
                "model_dir_name": model_dir_name,
                "display_model": _display_model_from_summary(summary, model_dir_name),
                "model_aliases": _model_aliases(summary, model_dir_name),
                "day": day,
                "summary_path": summary_path,
                "summary": summary,
            }
        )
    return pd.DataFrame(rows)


def infer_run_type(edge_csv: Path) -> str:
    parent_name = edge_csv.parent.name
    if parent_name in {"full_model", "selected_components"}:
        return parent_name

    filename = edge_csv.name
    if filename.startswith("full_model_edges_"):
        return "full_model"
    if filename.startswith("selected_components_edges_"):
        return "selected_components"
    return parent_name


def _pick_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _visualization_candidates(run_dir: Path, summary: dict) -> list[dict]:
    visuals = []
    if "results" in run_dir.parts:
        results_index = run_dir.parts.index("results")
        relative_to_results = Path(*run_dir.parts[results_index + 1 :])
        project_root = Path(*run_dir.parts[:results_index])
        image_dir = project_root / "results" / "images" / relative_to_results
        for path in sorted(image_dir.glob("*.png")):
            visuals.append({"label": path.stem, "kind": "png", "path": path})

    summary_visuals = summary.get("paths", {}).get("visualizations", {})
    preferred_names = (
        "layered_circuit",
        "budget_curve",
        "attention_head_scores",
        "mlp_layer_scores",
        "layer_flow",
    )
    for name in preferred_names:
        path = summary_visuals.get(name, {}).get("html")
        if path and Path(path).exists():
            visuals.append({"label": name, "kind": "html", "path": Path(path)})

    for path in sorted(run_dir.glob("*.html")):
        if any(item["path"] == path for item in visuals):
            continue
        if any(item["label"] == path.stem for item in visuals):
            continue
        visuals.append({"label": path.stem, "kind": "html", "path": path})

    return visuals


def parse_circuit_run_path(base: Path, edge_csv: Path) -> dict[str, str | None]:
    """Support both eap_ig/{model}/{day}/{run} and eap_ig/{model}/{dataset_set}/{day}/{run}."""
    relative_parts = edge_csv.parent.relative_to(base).parts
    if len(relative_parts) >= 4 and DATE_RE.fullmatch(relative_parts[-2]):
        return {
            "model_dir_name": relative_parts[0],
            "dataset_set": relative_parts[-3],
            "day": relative_parts[-2],
        }
    if len(relative_parts) >= 3 and DATE_RE.fullmatch(relative_parts[-2]):
        return {
            "model_dir_name": relative_parts[0],
            "dataset_set": None,
            "day": relative_parts[-2],
        }
    return {
        "model_dir_name": edge_csv.parent.parent.parent.name,
        "dataset_set": None,
        "day": edge_csv.parent.parent.name,
    }


def discover_circuit_runs(project_root: Path) -> pd.DataFrame:
    rows = []
    base = project_root / "results" / "eap_ig"
    for edge_csv in sorted(base.rglob("*_edges_*.csv")):
        filename = edge_csv.name
        if (
            "incoming_edges_" in filename
            or "_removed_edges_" in filename
            or "_nodes_" in filename
            or "_budget_sweep_" in filename
        ):
            continue

        run_type = infer_run_type(edge_csv)
        path_info = parse_circuit_run_path(base, edge_csv)
        day = path_info["day"]
        model_dir_name = path_info["model_dir_name"]
        summary_json = _pick_first_existing(
            [
                edge_csv.parent / f"{run_type}_summary_{day}.json",
                *_pick_globbed(edge_csv.parent, f"{run_type}_summary_*.json"),
                *_pick_globbed(edge_csv.parent, "*_summary_*.json"),
            ]
        )
        summary = safe_read_json(summary_json)

        node_csv = _pick_first_existing(
            [
                edge_csv.parent / f"{run_type}_nodes_{day}.csv",
                *_pick_globbed(edge_csv.parent, f"{run_type}_nodes_*.csv"),
                *_pick_globbed(edge_csv.parent, "*_nodes_*.csv"),
            ]
        )
        budget_csv = _pick_first_existing(
            [
                edge_csv.parent / f"{run_type}_budget_sweep_{day}.csv",
                *_pick_globbed(edge_csv.parent, f"{run_type}_budget_sweep_*.csv"),
                *_pick_globbed(edge_csv.parent, "*_budget_sweep_*.csv"),
            ]
        )

        rows.append(
            {
                "artifact_type": "eap_ig",
                "model_dir_name": model_dir_name,
                "display_model": _display_model_from_summary(summary, model_dir_name),
                "model_aliases": _model_aliases(summary, model_dir_name),
                "dataset_set": path_info["dataset_set"],
                "day": day,
                "run_type": run_type,
                "run_dir": edge_csv.parent,
                "edge_csv": edge_csv,
                "node_csv": node_csv,
                "budget_csv": budget_csv,
                "summary_json": summary_json,
                "summary": summary,
                "visualizations": _visualization_candidates(edge_csv.parent, summary),
            }
        )
    return pd.DataFrame(rows)


def _pick_globbed(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern))


def filter_artifacts(
    df: pd.DataFrame,
    model_query: str,
    day: str | None = None,
    dataset_set: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = df["model_aliases"].apply(lambda aliases: matches_model_query(model_query, aliases))
    out = df.loc[mask].copy()
    if day is not None:
        out = out.loc[out["day"] == day].copy()
    if dataset_set is not None and "dataset_set" in out.columns:
        out = out.loc[out["dataset_set"] == dataset_set].copy()
    return out.sort_values(["day"], ascending=[False]).reset_index(drop=True)


def normalize_model_spec(spec: str | dict) -> dict:
    if isinstance(spec, str):
        return {"model": spec, "date": None, "run_type": None, "dataset_set": None}
    return {
        "model": spec["model"],
        "date": spec.get("date"),
        "run_type": spec.get("run_type"),
        "dataset_set": spec.get("dataset_set"),
    }


def pick_latest_artifact(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values(["day"], ascending=[False]).iloc[0]


def _best_budget_key(row: pd.Series) -> tuple[float, float, float, float]:
    faithfulness = float(row.get("faithfulness_mean", float("-inf")))
    accuracy = float(row.get("accuracy_mean", float("-inf")))
    induced_node_count = float(row.get("induced_node_count", float("inf")))
    collapsed_edge_budget = float(row.get("collapsed_edge_budget", float("inf")))
    return (
        faithfulness,
        accuracy,
        -induced_node_count,
        -collapsed_edge_budget,
    )


def pick_best_budget_row(budget_df: pd.DataFrame) -> pd.Series | None:
    if budget_df is None or budget_df.empty:
        return None
    numeric = budget_df.copy()
    for column in (
        "collapsed_edge_budget",
        "expanded_edge_count",
        "induced_node_count",
        "faithfulness_mean",
        "faithfulness_std",
        "accuracy_mean",
        "accuracy_std",
        "validation_examples",
    ):
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    ranked = sorted(
        (row for _, row in numeric.iterrows()),
        key=_best_budget_key,
        reverse=True,
    )
    return ranked[0] if ranked else None


def _run_score(run_row: pd.Series) -> tuple[float, float, float, float]:
    budget_df = safe_read_csv(run_row.get("budget_csv"))
    best_budget = pick_best_budget_row(budget_df)
    if best_budget is None:
        return (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    return _best_budget_key(best_budget)


def pick_best_circuit_run(df: pd.DataFrame, requested_run_type: str | None = None) -> pd.Series | None:
    if df.empty:
        return None

    candidates = df.copy()
    if requested_run_type:
        candidates = candidates.loc[candidates["run_type"] == requested_run_type].copy()
        if candidates.empty:
            return None

    latest_day = candidates["day"].max()
    latest = candidates.loc[candidates["day"] == latest_day].copy()
    if len(latest) == 1:
        return latest.iloc[0]

    latest["run_score"] = latest.apply(_run_score, axis=1)
    latest = latest.sort_values(["run_score", "run_type"], ascending=[False, True])
    return latest.iloc[0]


def _parse_node(node: str) -> dict:
    node = str(node)
    if node == "input":
        return {"node": node, "kind": "input", "layer": -1, "head": np.nan}
    if node == "logits":
        return {"node": node, "kind": "logits", "layer": np.nan, "head": np.nan}

    attn_match = ATTN_RE.match(node)
    if attn_match:
        return {
            "node": node,
            "kind": "attention_head",
            "layer": int(attn_match.group(1)),
            "head": int(attn_match.group(2)),
        }

    mlp_match = MLP_RE.match(node)
    if mlp_match:
        return {
            "node": node,
            "kind": "mlp",
            "layer": int(mlp_match.group(1)),
            "head": np.nan,
        }

    return {"node": node, "kind": "other", "layer": np.nan, "head": np.nan}


def _fallback_node_scores_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["node", "induced_score"])
    parent = edges[["parent", "abs_score"]].rename(columns={"parent": "node", "abs_score": "induced_score"})
    child = edges[["child", "abs_score"]].rename(columns={"child": "node", "abs_score": "induced_score"})
    combined = pd.concat([parent, child], ignore_index=True)
    combined = combined.groupby("node", as_index=False)["induced_score"].sum()
    return combined.sort_values("induced_score", ascending=False).reset_index(drop=True)


def prepare_node_scores(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    base = nodes.copy() if nodes is not None and not nodes.empty else _fallback_node_scores_from_edges(edges)
    if base.empty:
        return pd.DataFrame(columns=["node", "induced_score", "kind", "layer", "head"])
    if "induced_score" not in base.columns and "abs_score" in base.columns:
        base = base.rename(columns={"abs_score": "induced_score"})
    base["induced_score"] = pd.to_numeric(base["induced_score"], errors="coerce")
    node_meta = pd.DataFrame([_parse_node(node) for node in base["node"]])
    for column in ("kind", "layer", "head"):
        if column not in base.columns:
            base[column] = node_meta[column].values
    out = base.reset_index(drop=True)
    return out.sort_values("induced_score", ascending=False).reset_index(drop=True)


def top_edges_table(edges: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    table = edges.copy()
    if "abs_score" in table.columns:
        table["abs_score"] = pd.to_numeric(table["abs_score"], errors="coerce")
        table = table.sort_values("abs_score", ascending=False)
    columns = [column for column in ("rank", "collapsed_edge", "parent", "child", "abs_score", "signed_sum", "underlying_edge_count") if column in table.columns]
    return table.loc[:, columns].head(top_n).reset_index(drop=True)


def top_component_tables(nodes: pd.DataFrame, edges: pd.DataFrame, top_n: int = 15) -> dict[str, pd.DataFrame]:
    scored = prepare_node_scores(nodes, edges)
    if scored.empty:
        return {"mlp": pd.DataFrame(), "attention_head": pd.DataFrame()}

    keep_columns = [column for column in ("node", "layer", "head", "induced_score", "rank") if column in scored.columns]
    mlp = scored.loc[scored["kind"] == "mlp", keep_columns].head(top_n).reset_index(drop=True)
    heads = scored.loc[scored["kind"] == "attention_head", keep_columns].head(top_n).reset_index(drop=True)
    return {"mlp": mlp, "attention_head": heads}


def budgeted_node_scores(edges: pd.DataFrame) -> pd.DataFrame:
    scores = _fallback_node_scores_from_edges(edges)
    if scores.empty:
        return pd.DataFrame(columns=["node", "induced_score", "rank", "kind", "layer", "head"])

    scores = scores.loc[~scores["node"].isin(["input", "logits"])].reset_index(drop=True)
    scores["rank"] = np.arange(1, len(scores) + 1)
    node_meta = pd.DataFrame([_parse_node(node) for node in scores["node"]])
    return pd.concat([scores, node_meta.drop(columns=["node"])], axis=1)


def budgeted_circuit_view(
    artifact: dict,
    budget: int | None = None,
    *,
    top_edge_rows: int = 15,
    top_component_rows: int = 15,
) -> dict:
    edges = (artifact or {}).get("edges")
    if edges is None or edges.empty:
        raise ValueError("Circuit artifact is missing ranked edges.")

    max_budget = int(len(edges))
    if budget is None:
        best_budget = (artifact or {}).get("best_budget")
        resolved_budget = int(best_budget["collapsed_edge_budget"]) if best_budget is not None else max_budget
    else:
        resolved_budget = int(budget)

    if resolved_budget <= 0 or resolved_budget > max_budget:
        raise ValueError(
            f"Budget must be between 1 and {max_budget}, got {resolved_budget}."
        )

    budget_edges = edges.head(resolved_budget).copy().reset_index(drop=True)
    if "rank" not in budget_edges.columns:
        budget_edges["rank"] = np.arange(1, len(budget_edges) + 1)

    budget_nodes = budgeted_node_scores(budget_edges)
    component_tables = top_component_tables(
        budget_nodes,
        budget_edges,
        top_n=max(top_component_rows, 1),
    )

    evaluated_budget_row = None
    budget_frame = (artifact or {}).get("budget")
    if budget_frame is not None and not budget_frame.empty and "collapsed_edge_budget" in budget_frame.columns:
        matches = budget_frame.loc[
            pd.to_numeric(budget_frame["collapsed_edge_budget"], errors="coerce") == resolved_budget
        ]
        if not matches.empty:
            evaluated_budget_row = matches.iloc[0]

    return {
        "collapsed_edge_budget": resolved_budget,
        "max_budget": max_budget,
        "is_evaluated_budget": evaluated_budget_row is not None,
        "evaluated_budget_row": evaluated_budget_row,
        "edges": budget_edges,
        "nodes": budget_nodes,
        "top_edges": top_edges_table(budget_edges, top_n=max(top_edge_rows, 1)),
        "top_mlps": component_tables["mlp"].head(top_component_rows).reset_index(drop=True),
        "top_attention_heads": component_tables["attention_head"].head(top_component_rows).reset_index(drop=True),
    }


def _plot_node_metadata(node_name: str) -> dict[str, Any]:
    if node_name == "input":
        return {"kind": "input", "layer": -1, "head": None}
    if node_name == "logits":
        return {"kind": "logits", "layer": -1, "head": None}

    parsed = _parse_node(node_name)
    kind = parsed["kind"]
    if kind == "attention_head":
        kind = "attn"
    if kind not in {"mlp", "attn"}:
        kind = "other"
    return {
        "kind": kind,
        "layer": int(parsed["layer"]) if not pd.isna(parsed["layer"]) else 0,
        "head": None if pd.isna(parsed["head"]) else int(parsed["head"]),
    }


def _plot_node_sort_key(node_name: str) -> tuple[int, int, int, str]:
    meta = _plot_node_metadata(node_name)
    type_order = {"input": 0, "attn": 1, "mlp": 2, "logits": 3, "other": 4}[meta["kind"]]
    head_order = -1 if meta["head"] is None else int(meta["head"])
    return (int(meta["layer"]), type_order, head_order, node_name)


def _residual_checkpoint_sequence(max_layer: int) -> list[str]:
    checkpoints = ["resid_pre_0"]
    for layer in range(max_layer + 1):
        checkpoints.append(f"resid_mid_{layer}")
        checkpoints.append(f"resid_post_{layer}")
    return checkpoints


def _residual_checkpoint_label(checkpoint: str) -> str:
    if checkpoint == "resid_pre_0":
        return "pre0"
    if checkpoint.startswith("resid_mid_"):
        return f"mid{checkpoint.removeprefix('resid_mid_')}"
    if checkpoint.startswith("resid_post_"):
        return f"post{checkpoint.removeprefix('resid_post_')}"
    return checkpoint


def _component_output_checkpoint(node_name: str) -> str | None:
    meta = _plot_node_metadata(node_name)
    kind = meta["kind"]
    if kind == "input":
        return "resid_pre_0"
    if kind == "attn":
        return f"resid_mid_{int(meta['layer'])}"
    if kind == "mlp":
        return f"resid_post_{int(meta['layer'])}"
    return None


def _component_input_checkpoint(node_name: str, max_layer: int) -> str | None:
    meta = _plot_node_metadata(node_name)
    kind = meta["kind"]
    if kind == "attn":
        layer = int(meta["layer"])
        return "resid_pre_0" if layer == 0 else f"resid_post_{layer - 1}"
    if kind == "mlp":
        return f"resid_mid_{int(meta['layer'])}"
    if kind == "logits":
        return f"resid_post_{max_layer}"
    return None


def _aggregate_residual_path_edges(
    edge_frame: pd.DataFrame,
    *,
    normalize_by_path_length: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, tuple[float, float]], list[str]] | None:
    if edge_frame.empty:
        return None

    component_nodes = sorted(
        set(edge_frame["parent"]).union(edge_frame["child"]),
        key=_plot_node_sort_key,
    )
    layer_values = [
        int(meta["layer"])
        for meta in (_plot_node_metadata(node) for node in component_nodes)
        if meta["kind"] in {"attn", "mlp"}
    ]
    if not layer_values:
        return None

    max_layer = max(layer_values)
    checkpoints = _residual_checkpoint_sequence(max_layer)
    checkpoint_index = {name: idx for idx, name in enumerate(checkpoints)}

    positions: dict[str, tuple[float, float]] = {
        checkpoint: (float(idx), 0.0)
        for idx, checkpoint in enumerate(checkpoints)
    }

    scored_nodes = prepare_node_scores(pd.DataFrame(), edge_frame)
    score_map = {
        str(row.node): float(row.induced_score)
        for row in scored_nodes.itertuples(index=False)
    }

    attn_by_layer: dict[int, list[str]] = defaultdict(list)
    mlp_by_layer: dict[int, list[str]] = defaultdict(list)
    for node in component_nodes:
        meta = _plot_node_metadata(node)
        if meta["kind"] == "attn":
            attn_by_layer[int(meta["layer"])].append(node)
        elif meta["kind"] == "mlp":
            mlp_by_layer[int(meta["layer"])].append(node)

    for layer, nodes in attn_by_layer.items():
        nodes.sort(key=lambda node: _plot_node_sort_key(node))
        center = (len(nodes) - 1) / 2.0
        x = float(2 * layer) + 0.5
        for idx, node in enumerate(nodes):
            positions[node] = (x, 2.2 + 0.75 * (center - idx))

    for layer, nodes in mlp_by_layer.items():
        nodes.sort(key=lambda node: _plot_node_sort_key(node))
        center = (len(nodes) - 1) / 2.0
        x = float(2 * layer) + 1.5
        for idx, node in enumerate(nodes):
            positions[node] = (x, -2.2 - 0.75 * (center - idx))

    if "input" in component_nodes:
        positions["input"] = (-0.75, 1.1)
    if "logits" in component_nodes:
        positions["logits"] = (float(len(checkpoints) - 1) + 0.75, 1.1)

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    residual_load: dict[str, float] = defaultdict(float)

    for row in edge_frame.itertuples(index=False):
        parent = str(row.parent)
        child = str(row.child)
        abs_score = float(row.abs_score)
        signed_sum = float(getattr(row, "signed_sum", 0.0))
        collapsed_edge = str(getattr(row, "collapsed_edge", f"{parent}->{child}"))

        source_checkpoint = _component_output_checkpoint(parent)
        target_checkpoint = _component_input_checkpoint(child, max_layer=max_layer)
        if source_checkpoint is None or target_checkpoint is None:
            continue

        source_index = checkpoint_index[source_checkpoint]
        target_index = checkpoint_index[target_checkpoint]

        route_edges: list[tuple[str, str, str]] = [(parent, source_checkpoint, "write")]
        if source_index <= target_index:
            for checkpoint_pos in range(source_index, target_index):
                route_edges.append(
                    (
                        checkpoints[checkpoint_pos],
                        checkpoints[checkpoint_pos + 1],
                        "carry",
                    )
                )
        elif source_checkpoint != target_checkpoint:
            route_edges.append((source_checkpoint, target_checkpoint, "carry"))
        route_edges.append((target_checkpoint, child, "read"))

        path_length = len(route_edges)
        if path_length <= 0:
            continue

        weight = abs_score / path_length if normalize_by_path_length else abs_score
        signed_weight = signed_sum / path_length if normalize_by_path_length else signed_sum

        for src, dst, segment_kind in route_edges:
            segment = aggregated.setdefault(
                (src, dst),
                {
                    "source": src,
                    "target": dst,
                    "kind": segment_kind,
                    "load": 0.0,
                    "signed_load": 0.0,
                    "collapsed_edge_count": 0,
                    "collapsed_edge_examples": [],
                    "path_length_total": 0,
                },
            )
            segment["load"] += weight
            segment["signed_load"] += signed_weight
            segment["collapsed_edge_count"] += 1
            segment["path_length_total"] += path_length
            if len(segment["collapsed_edge_examples"]) < 4:
                segment["collapsed_edge_examples"].append(collapsed_edge)

            residual_load[src] += weight
            residual_load[dst] += weight

    if not aggregated:
        return None

    aggregated_edges = sorted(aggregated.values(), key=lambda item: item["load"])
    return aggregated_edges, score_map, positions, checkpoints


def _build_residualized_circuit_figure(
    edge_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    *,
    normalize_by_path_length: bool = False,
    min_height: int = 650,
):
    import plotly.express as px
    import plotly.graph_objects as go

    aggregated = _aggregate_residual_path_edges(
        edge_frame,
        normalize_by_path_length=normalize_by_path_length,
    )
    if aggregated is None:
        return None

    aggregated_edges, score_map, positions, checkpoints = aggregated
    residual_nodes = set(checkpoints)
    component_nodes = [node for node in positions if node not in residual_nodes]

    residual_node_load: dict[str, float] = defaultdict(float)
    for edge in aggregated_edges:
        residual_node_load[str(edge["source"])] += float(edge["load"])
        residual_node_load[str(edge["target"])] += float(edge["load"])

    component_scores = prepare_node_scores(node_frame, edge_frame)
    if component_scores.empty:
        return None
    component_score_map = {
        str(row.node): float(row.induced_score)
        for row in component_scores.itertuples(index=False)
    }

    max_component_score = max(component_score_map.values()) if component_score_map else 1.0
    max_residual_load = max(residual_node_load.values()) if residual_node_load else 1.0
    max_edge_load = max(float(edge["load"]) for edge in aggregated_edges) if aggregated_edges else 1.0

    color_scale = px.colors.sequential.Blues

    color_map = {
        "input": "#7f7f7f",
        "attn": "#1f77b4",
        "mlp": "#ff7f0e",
        "logits": "#2f2f2f",
        "other": "#9467bd",
    }

    fig = go.Figure()
    for edge in aggregated_edges:
        source = str(edge["source"])
        target = str(edge["target"])
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        load_fraction = float(edge["load"]) / max_edge_load if max_edge_load else 0.0
        line_color = px.colors.sample_colorscale(color_scale, [max(0.15, load_fraction)])[0]
        line_width = 1.5 + 7.5 * load_fraction
        hover_text = (
            f"{source} -> {target}<br>"
            f"Segment kind: {edge['kind']}<br>"
            f"Aggregated load: {float(edge['load']):.4f}<br>"
            f"Aggregated signed load: {float(edge['signed_load']):.4f}<br>"
            f"Routed collapsed edges: {int(edge['collapsed_edge_count'])}<br>"
            f"Example contributors: {', '.join(edge['collapsed_edge_examples'])}"
        )
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=line_color, width=line_width),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
            )
        )

    checkpoint_x = [positions[checkpoint][0] for checkpoint in checkpoints]
    checkpoint_y = [positions[checkpoint][1] for checkpoint in checkpoints]
    checkpoint_sizes = [
        10.0 + 10.0 * (residual_node_load[checkpoint] / max_residual_load if max_residual_load else 0.0)
        for checkpoint in checkpoints
    ]
    checkpoint_hover = [
        (
            f"Checkpoint: {_residual_checkpoint_label(checkpoint)}<br>"
            f"Incident routed load: {residual_node_load[checkpoint]:.4f}"
        )
        for checkpoint in checkpoints
    ]
    fig.add_trace(
        go.Scatter(
            x=checkpoint_x,
            y=checkpoint_y,
            mode="markers",
            marker=dict(size=checkpoint_sizes, color="#7f7f7f", line=dict(color="white", width=1.0)),
            customdata=checkpoint_hover,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        )
    )

    component_x: list[float] = []
    component_y: list[float] = []
    component_text: list[str] = []
    component_sizes: list[float] = []
    component_colors: list[str] = []
    component_hover: list[str] = []

    for node in sorted(component_nodes, key=_plot_node_sort_key):
        meta = _plot_node_metadata(node)
        if node not in component_score_map and meta["kind"] not in {"input", "logits"}:
            continue
        x, y = positions[node]
        induced_score = component_score_map.get(node, 0.0)
        component_x.append(x)
        component_y.append(y)
        component_text.append(node)
        if meta["kind"] in {"input", "logits"}:
            component_sizes.append(14.0)
        else:
            component_sizes.append(
                14.0 + 20.0 * (induced_score / max_component_score if max_component_score else 0.0)
            )
        component_colors.append(color_map.get(str(meta["kind"]), color_map["other"]))
        induced_text = f"{induced_score:.4f}" if meta["kind"] not in {"input", "logits"} else "N/A"
        component_hover.append(
            "<br>".join(
                [
                    f"Node: {node}",
                    f"Kind: {meta['kind']}",
                    f"Layer: {meta['layer']}" if meta["kind"] not in {"input", "logits"} else "Layer: N/A",
                    f"Induced score: {induced_text}",
                    f"Fallback routed load: {score_map.get(node, 0.0):.4f}",
                ]
            )
        )

    fig.add_trace(
        go.Scatter(
            x=component_x,
            y=component_y,
            mode="markers+text",
            text=component_text,
            textposition="top center",
            marker=dict(size=component_sizes, color=component_colors, line=dict(color="white", width=1.2)),
            customdata=component_hover,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        )
    )

    tickvals = [positions[checkpoint][0] for checkpoint in checkpoints]
    ticktext = [_residual_checkpoint_label(checkpoint) for checkpoint in checkpoints]
    view_label = "path-length normalized" if normalize_by_path_length else "raw"

    fig.update_layout(
        title=f"Residualized full-circuit view ({view_label} routed load)",
        template="plotly_white",
        height=max(int(min_height), 700),
        hovermode="closest",
        margin=dict(l=40, r=40, t=90, b=60),
    )
    fig.update_xaxes(
        title="Residual-stream checkpoints across the forward pass",
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        title="Components around the residual carrier",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    fig.add_annotation(
        x=0.5,
        y=1.09,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            "Edges route each collapsed EAP dependency through residual-stream checkpoints. "
            "Color and width show aggregated routed load; component node size shows induced score."
        ),
    )
    return fig


def _build_layered_circuit_figure(
    edge_frame: pd.DataFrame,
    top_k: int = 40,
    *,
    min_height: int = 550,
    height_per_node: int = 32,
):
    import plotly.graph_objects as go

    if edge_frame.empty:
        raise ValueError("edge_frame is empty. Run EAP attribution before visualizing the circuit.")

    top_edges = edge_frame.nlargest(top_k, "abs_score").copy()
    nodes = sorted(set(top_edges["parent"]).union(top_edges["child"]), key=_plot_node_sort_key)
    metadata = {node: _plot_node_metadata(node) for node in nodes}

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
        bucket_key = node_meta["kind"] if node_meta["kind"] in {"input", "logits"} else int(node_meta["layer"])
        buckets[bucket_key].append(node)

    positions: dict[str, tuple[float, float]] = {}
    for bucket_nodes in buckets.values():
        bucket_nodes.sort(key=_plot_node_sort_key)
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
        edge_color = "rgba(31,119,180,0.70)" if float(row.signed_sum) >= 0 else "rgba(214,39,40,0.70)"
        edge_width = 1.5 + 8.0 * (float(row.abs_score) / max_edge_score if max_edge_score else 0.0)
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
        node_size.append(14 + 18 * (incident_strength[node] / max_incident if max_incident else 0.0))
        node_hover.append(
            "<br>".join(
                [
                    f"Node: {node}",
                    f"Kind: {node_meta['kind']}",
                    f"Layer: {node_meta['layer']}" if node_meta["kind"] not in {"input", "logits"} else "Layer: N/A",
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
            marker=dict(size=node_size, color=node_color, line=dict(color="white", width=1.2)),
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
        height=max(int(min_height), int(height_per_node) * len(nodes)),
        hovermode="closest",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(title="Model depth", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False)
    fig.update_yaxes(title="Components within layer", showgrid=False, zeroline=False, showticklabels=False)
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


def _build_attention_head_score_figure(node_frame: pd.DataFrame):
    import plotly.express as px

    attn_rows: list[dict[str, Any]] = []
    for row in node_frame.itertuples(index=False):
        meta = _plot_node_metadata(str(row.node))
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
            customdata[row_idx, col_idx] = int(match["rank"].iloc[0]) if not match.empty else None

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


def _build_mlp_layer_score_figure(node_frame: pd.DataFrame):
    import plotly.graph_objects as go

    mlp_rows: list[dict[str, Any]] = []
    for row in node_frame.itertuples(index=False):
        meta = _plot_node_metadata(str(row.node))
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


def _build_component_score_heatmap_figure(node_frame: pd.DataFrame):
    import plotly.graph_objects as go

    rows: list[dict[str, Any]] = []
    for row in node_frame.itertuples(index=False):
        meta = _plot_node_metadata(str(row.node))
        if meta["kind"] == "attn":
            rows.append(
                {
                    "component_label": f"H{int(meta['head'])}",
                    "component_sort": int(meta["head"]),
                    "layer": int(meta["layer"]),
                    "induced_score": float(row.induced_score),
                    "rank": int(row.rank),
                    "node": str(row.node),
                }
            )
        elif meta["kind"] == "mlp":
            rows.append(
                {
                    "component_label": "MLP",
                    "component_sort": 10_000,
                    "layer": int(meta["layer"]),
                    "induced_score": float(row.induced_score),
                    "rank": int(row.rank),
                    "node": str(row.node),
                }
            )
    if not rows:
        return None

    frame = pd.DataFrame(rows)
    heatmap = (
        frame.pivot(index="component_label", columns="layer", values="induced_score")
        .sort_index(axis=1)
    )
    row_order = (
        frame[["component_label", "component_sort"]]
        .drop_duplicates()
        .sort_values(["component_sort", "component_label"], kind="stable")["component_label"]
        .tolist()
    )
    heatmap = heatmap.reindex(index=row_order).fillna(0.0)

    raw_values = heatmap.to_numpy(dtype=float)
    zmax = float(np.nanmax(raw_values)) if raw_values.size else 0.0
    zmax = zmax if zmax > 0 else 1.0

    customdata = np.empty((heatmap.shape[0], heatmap.shape[1]), dtype=object)
    for row_idx, component_label in enumerate(heatmap.index.tolist()):
        for col_idx, layer in enumerate(heatmap.columns.tolist()):
            match = frame[
                (frame["component_label"] == component_label) & (frame["layer"] == layer)
            ]
            customdata[row_idx, col_idx] = (
                {
                    "node": str(match["node"].iloc[0]),
                    "rank": int(match["rank"].iloc[0]),
                    "induced_score": float(match["induced_score"].iloc[0]),
                }
                if not match.empty
                else None
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=raw_values,
            x=[f"L{int(layer)}" for layer in heatmap.columns.tolist()],
            y=heatmap.index.tolist(),
            customdata=customdata,
            colorscale="Blues",
            zmin=0.0,
            zmax=zmax,
            colorbar=dict(title="induced score"),
            hovertemplate=(
                "Layer=%{x}<br>"
                "Component=%{y}<br>"
                "Node=%{customdata.node}<br>"
                "Induced score=%{customdata.induced_score:.4f}<br>"
                "Node rank=%{customdata.rank}<br>"
                "Heatmap value=%{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Induced score heatmap for heads and MLPs",
        template="plotly_white",
        xaxis_title="Layer",
        yaxis_title="Component",
        height=max(520, 26 * max(len(heatmap.index), 1) + 180),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    return fig


def _build_layer_flow_figure(edge_frame: pd.DataFrame):
    import plotly.express as px

    if edge_frame.empty:
        return None

    flow_rows: list[dict[str, Any]] = []
    for row in edge_frame.itertuples(index=False):
        parent_meta = _plot_node_metadata(str(row.parent))
        child_meta = _plot_node_metadata(str(row.child))
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
        labels={"x": "Child layer", "y": "Parent layer", "color": "Summed |edge score|"},
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


def build_live_budgeted_circuit_figures(
    edge_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    *,
    top_k_edges: int = 40,
    layered_circuit_min_height: int = 550,
    layered_circuit_height_per_node: int = 32,
) -> dict[str, Any]:
    top_k = min(max(int(top_k_edges), 1), len(edge_frame)) if not edge_frame.empty else 0
    return {
        "layered_circuit": (
            _build_layered_circuit_figure(
                edge_frame,
                top_k=top_k,
                min_height=layered_circuit_min_height,
                height_per_node=layered_circuit_height_per_node,
            )
            if top_k > 0
            else None
        ),
        "residualized_normalized": _build_residualized_circuit_figure(
            edge_frame,
            node_frame,
            normalize_by_path_length=True,
        ),
        "layer_flow": _build_layer_flow_figure(edge_frame),
        "component_score_heatmap": _build_component_score_heatmap_figure(node_frame),
        "attention_head_scores": _build_attention_head_score_figure(node_frame),
        "mlp_layer_scores": _build_mlp_layer_score_figure(node_frame),
    }


def discover_manual_images(project_root: Path, model_dir_name: str, day: str | None = None) -> list[dict]:
    visuals = []
    base = project_root / "results" / "images" / "manual_circuit_discovery" / model_dir_name
    candidates = []
    if day is not None:
        candidates.append(base / day / "plots")
    candidates.append(base / "undated" / "plots")

    for directory in candidates:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.png")):
            visuals.append({"label": path.stem, "kind": "png", "path": path})
    return visuals


def summarize_diagnostic(summary: dict) -> dict:
    dataset_summary = summary.get("dataset_summary", {})
    alignment = summary.get("tokenization_diagnostics", {}).get("raw_dataset_alignment", {})
    targets = summary.get("tokenization_diagnostics", {}).get("target_sets", {})

    source_success_count = int(dataset_summary.get("source_model_success_count", 0) or 0)
    scored_on_source = int(dataset_summary.get("target_on_source_scored_count", 0) or 0)
    filtered_count = int(dataset_summary.get("target_filtered_count", 0) or 0)
    total_pairs = int(alignment.get("total_pairs", 0) or 0)
    fully_aligned = int(alignment.get("fully_aligned", 0) or 0)

    animate = targets.get("animate", {})
    inanimate = targets.get("inanimate", {})

    return {
        "target_model": dataset_summary.get("target_model_requested")
        or dataset_summary.get("target_model")
        or summary.get("config", {}).get("model_name"),
        "gpt2_filtered_pair_accuracy": dataset_summary.get("target_on_source_accuracy", {}).get("pair_success", {}).get("rate"),
        "gpt2_filtered_pair_accuracy_count": dataset_summary.get("target_on_source_accuracy", {}).get("pair_success", {}).get("count"),
        "gpt2_filtered_example_count": dataset_summary.get("target_on_source_accuracy", {}).get("example_count"),
        "raw_pair_accuracy": dataset_summary.get("target_raw_accuracy", {}).get("pair_success", {}).get("rate"),
        "raw_pair_accuracy_count": dataset_summary.get("target_raw_accuracy", {}).get("pair_success", {}).get("count"),
        "raw_example_count": dataset_summary.get("target_raw_accuracy", {}).get("example_count"),
        "source_success_count": source_success_count,
        "scored_on_source_count": scored_on_source,
        "mp_token_mismatched": max(source_success_count - scored_on_source, 0),
        "filtered_for_experiments_count": filtered_count,
        "mp_metric_failure": max(scored_on_source - filtered_count, 0),
        # Backward-compatible aliases. These are kept so older notebook code can still
        # resolve the same keys, but the new names above are the intended labels.
        "ignored_on_source_count": max(source_success_count - scored_on_source, 0),
        "ignored_for_experiments_count": max(source_success_count - filtered_count, 0),
        "total_pairs": total_pairs,
        "fully_aligned_pairs": fully_aligned,
        "ignored_pairs": max(total_pairs - fully_aligned, 0),
        "metadata_missing": int(alignment.get("metadata_missing", 0) or 0),
        "sequence_length_mismatch": int(alignment.get("sequence_length_mismatch", 0) or 0),
        "patient_span_misaligned": int(alignment.get("patient_span_misaligned", 0) or 0),
        "verb_span_misaligned": int(alignment.get("verb_span_misaligned", 0) or 0),
        "animate_ignored_targets": max(int(animate.get("total", 0) or 0) - int(animate.get("single_token_count", 0) or 0), 0),
        "animate_total": int(animate.get("total", 0) or 0),
        "inanimate_ignored_targets": max(int(inanimate.get("total", 0) or 0) - int(inanimate.get("single_token_count", 0) or 0), 0),
        "inanimate_total": int(inanimate.get("total", 0) or 0),
    }


def format_diagnostic_recap(metrics: dict) -> str:
    if not metrics:
        return "No diagnostic summary available."
    accuracy = metrics.get("gpt2_filtered_pair_accuracy")
    count = metrics.get("gpt2_filtered_pair_accuracy_count")
    example_count = metrics.get("gpt2_filtered_example_count")
    accuracy_text = "n/a"
    if accuracy is not None and count is not None and example_count is not None:
        accuracy_text = f"{accuracy:.4f} ({count}/{example_count})"

    return (
        f"{metrics['target_model']}: gpt2-filtered accuracy {accuracy_text}; "
        f"mp_token_mismatched {metrics['mp_token_mismatched']}; "
        f"mp_metric_failure {metrics['mp_metric_failure']}; "
        f"sentence pairs ignored {metrics['ignored_pairs']} "
        f"(missing metadata {metrics['metadata_missing']}, "
        f"seq-len mismatch {metrics['sequence_length_mismatch']}, "
        f"patient mismatch {metrics['patient_span_misaligned']}, "
        f"verb mismatch {metrics['verb_span_misaligned']}); "
        f"targets ignored animate {metrics['animate_ignored_targets']}/{metrics['animate_total']}, "
        f"inanimate {metrics['inanimate_ignored_targets']}/{metrics['inanimate_total']}."
    )


def load_circuit_artifact(run_row: pd.Series, project_root: Path) -> dict:
    edges = safe_read_csv(run_row.get("edge_csv"))
    nodes = safe_read_csv(run_row.get("node_csv"))
    budget = safe_read_csv(run_row.get("budget_csv"))
    summary = run_row.get("summary") or safe_read_json(run_row.get("summary_json"))

    if "abs_score" in edges.columns:
        edges["abs_score"] = pd.to_numeric(edges["abs_score"], errors="coerce")
        edges = edges.sort_values("abs_score", ascending=False).reset_index(drop=True)

    best_budget = pick_best_budget_row(budget)
    component_tables = top_component_tables(nodes, edges)

    visuals = list(run_row.get("visualizations") or [])
    for visual in discover_manual_images(project_root, run_row["model_dir_name"], run_row["day"]):
        if any(existing["path"] == visual["path"] for existing in visuals):
            continue
        visuals.append(visual)

    return {
        "meta": run_row,
        "summary": summary,
        "edges": edges,
        "nodes": nodes,
        "budget": budget,
        "best_budget": best_budget,
        "visualizations": visuals,
        "top_edges": top_edges_table(edges),
        "top_mlps": component_tables["mlp"],
        "top_attention_heads": component_tables["attention_head"],
    }


def resolve_model_report(
    project_root: Path,
    spec: str | dict,
    diagnostics_df: pd.DataFrame,
    circuits_df: pd.DataFrame,
) -> dict:
    normalized = normalize_model_spec(spec)

    diagnostic_candidates = filter_artifacts(diagnostics_df, normalized["model"], normalized["date"])
    diagnostic_row = pick_latest_artifact(diagnostic_candidates)
    diagnostic_summary = diagnostic_row["summary"] if diagnostic_row is not None else {}
    diagnostic_metrics = summarize_diagnostic(diagnostic_summary) if diagnostic_summary else {}

    circuit_candidates = filter_artifacts(
        circuits_df,
        normalized["model"],
        normalized["date"],
        normalized["dataset_set"],
    )
    chosen_run = pick_best_circuit_run(circuit_candidates, requested_run_type=normalized["run_type"])
    circuit_artifact = load_circuit_artifact(chosen_run, project_root) if chosen_run is not None else None

    display_model = (
        (circuit_artifact["meta"]["display_model"] if circuit_artifact is not None else None)
        or (diagnostic_row["display_model"] if diagnostic_row is not None else None)
        or normalized["model"]
    )

    availability = {
        "model_query": normalized["model"],
        "display_model": display_model,
        "requested_date": normalized["date"],
        "diagnostic_available": diagnostic_row is not None,
        "diagnostic_day": diagnostic_row["day"] if diagnostic_row is not None else None,
        "circuit_available": chosen_run is not None,
        "circuit_day": chosen_run["day"] if chosen_run is not None else None,
        "circuit_run_type": chosen_run["run_type"] if chosen_run is not None else None,
        "visualization_count": len(circuit_artifact["visualizations"]) if circuit_artifact is not None else 0,
    }

    if circuit_artifact is not None and circuit_artifact["best_budget"] is not None:
        best_budget = circuit_artifact["best_budget"]
        availability.update(
            {
                "best_budget": int(best_budget["collapsed_edge_budget"]),
                "best_budget_percent": float(best_budget["budget_fraction"] * 100.0)
                if "budget_fraction" in best_budget.index
                else np.nan,
                "best_faithfulness": float(best_budget["faithfulness_mean"]),
                "best_accuracy": float(best_budget["accuracy_mean"]) if "accuracy_mean" in best_budget.index else np.nan,
                "best_induced_node_count": int(best_budget["induced_node_count"]),
            }
        )

    return {
        "spec": normalized,
        "display_model": display_model,
        "diagnostic_row": diagnostic_row,
        "diagnostic_metrics": diagnostic_metrics,
        "diagnostic_recap": format_diagnostic_recap(diagnostic_metrics) if diagnostic_metrics else "No diagnostic summary available.",
        "circuit_run_row": chosen_run,
        "circuit_artifact": circuit_artifact,
        "availability": availability,
    }


def build_model_reports(project_root: Path, model_specs: list[str | dict]) -> tuple[pd.DataFrame, list[dict]]:
    diagnostics_df = discover_model_diagnostics(project_root)
    circuits_df = discover_circuit_runs(project_root)
    reports = [
        resolve_model_report(
            project_root=project_root,
            spec=spec,
            diagnostics_df=diagnostics_df,
            circuits_df=circuits_df,
        )
        for spec in model_specs
    ]
    availability = pd.DataFrame([report["availability"] for report in reports])
    return availability, reports


def budget_curve_frame(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        artifact = report.get("circuit_artifact")
        if artifact is None or artifact["budget"].empty:
            continue
        budget = artifact["budget"].copy()
        budget["display_model"] = report["display_model"]
        budget["run_type"] = artifact["meta"]["run_type"]
        budget["day"] = artifact["meta"]["day"]
        rows.append(budget)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def circuit_summary_frame(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        artifact = report.get("circuit_artifact")
        if artifact is None:
            continue
        best_budget = artifact.get("best_budget")
        row = {
            "display_model": report["display_model"],
            "run_type": artifact["meta"]["run_type"],
            "day": artifact["meta"]["day"],
            "n_edges": int(len(artifact["edges"])),
            "n_nodes": int(len(artifact["nodes"])),
            "visualization_count": len(artifact["visualizations"]),
        }
        if best_budget is not None:
            row.update(
                {
                    "best_budget": int(best_budget["collapsed_edge_budget"]),
                    "best_budget_percent": float(best_budget["budget_fraction"] * 100.0)
                    if "budget_fraction" in best_budget.index
                    else np.nan,
                    "best_faithfulness": float(best_budget["faithfulness_mean"]),
                    "best_accuracy": float(best_budget["accuracy_mean"]) if "accuracy_mean" in best_budget.index else np.nan,
                    "best_induced_node_count": int(best_budget["induced_node_count"]),
                    "validation_examples": int(best_budget["validation_examples"]) if "validation_examples" in best_budget.index else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostic_summary_frame(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        metrics = report.get("diagnostic_metrics")
        if not metrics:
            continue
        rows.append(
            {
                "display_model": report["display_model"],
                "diagnostic_day": report["diagnostic_row"]["day"] if report["diagnostic_row"] is not None else None,
                "gpt2_filtered_accuracy": metrics["gpt2_filtered_pair_accuracy"],
                "gpt2_filtered_count": metrics["gpt2_filtered_pair_accuracy_count"],
                "gpt2_filtered_examples": metrics["gpt2_filtered_example_count"],
                "mp_token_mismatched": metrics["mp_token_mismatched"],
                "mp_metric_failure": metrics["mp_metric_failure"],
                "ignored_pairs": metrics["ignored_pairs"],
                "animate_ignored_targets": metrics["animate_ignored_targets"],
                "inanimate_ignored_targets": metrics["inanimate_ignored_targets"],
            }
        )
    return pd.DataFrame(rows)

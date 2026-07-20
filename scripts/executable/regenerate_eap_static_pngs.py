from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from circuit_finder_paths import ensure_dir, resolve_animacy_circuit_root
from utils import canonical_model_name, safe_model_name, save_json


ATTN_RE = re.compile(r"^a(\d+)[._]h(\d+)$")
MLP_RE = re.compile(r"^m(\d+)$")

PIPELINE_STEMS = {
    "full_model": "full_model",
    "selected_components": "selected_components",
}
RELATIVE_BUDGET_TICKVALS = (0.005, *tuple(value / 100.0 for value in range(1, 11)))
RELATIVE_BUDGET_TICKTEXT = ("0.5%", *tuple(f"{value}%" for value in range(1, 11)))
MODEL_SPECIFIC_CORRECT = "model_specific_correct"
DATASET_SET_NAMES = (MODEL_SPECIFIC_CORRECT, "shared_correct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate static Matplotlib PNGs from saved EAP ranking CSVs."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--day", required=True)
    parser.add_argument("--pipeline", choices=tuple(PIPELINE_STEMS), default="full_model")
    parser.add_argument("--dataset-set", default=MODEL_SPECIFIC_CORRECT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k-edges", type=int, default=40)
    parser.add_argument("--start-path", type=Path, default=None)
    return parser.parse_args()


def default_output_dir(
    project_root: Path,
    model_name: str,
    day: str,
    pipeline: str,
    dataset_set: str,
) -> Path:
    if pipeline == "full_model":
        return ensure_dir(
            project_root
            / "results"
            / "eap_ig"
            / safe_model_name(canonical_model_name(model_name))
            / dataset_set
            / day
            / "full_model"
        )
    return ensure_dir(
        project_root
        / "results"
        / "eap_ig"
        / safe_model_name(canonical_model_name(model_name))
        / day
        / "selected_components"
    )


def resolve_image_output_dir(project_root: Path, file_path: Path) -> Path:
    results_root = (project_root / "results").resolve()
    image_root = ensure_dir(results_root / "images")
    try:
        relative = file_path.resolve().relative_to(results_root)
    except ValueError:
        return ensure_dir(image_root / "legacy" / "global")

    parts = relative.parts
    if len(parts) >= 4 and parts[0] == "eap_ig":
        destination = image_root / "eap_ig" / parts[1]
        part_index = 2
        if parts[2] in DATASET_SET_NAMES:
            destination = destination / parts[2] / parts[3]
            part_index = 4
        else:
            destination /= parts[2]
            part_index = 3
        if len(parts) > part_index and parts[part_index] in {"full_model", "selected_components", "comparison"}:
            destination /= parts[part_index]
        return ensure_dir(destination)

    return ensure_dir(image_root / "legacy" / "global")


def require_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def node_kind(node: str) -> str:
    if ATTN_RE.match(str(node)):
        return "attention"
    if MLP_RE.match(str(node)):
        return "mlp"
    return str(node)


def node_layer(node: str, max_layer: int | None = None) -> int | None:
    node = str(node)
    if node == "input":
        return -1
    if node == "logits":
        return None if max_layer is None else max_layer + 1
    match = ATTN_RE.match(node)
    if match:
        return int(match.group(1))
    match = MLP_RE.match(node)
    if match:
        return int(match.group(1))
    return None


def max_model_layer(edges: pd.DataFrame, nodes: pd.DataFrame) -> int:
    values: list[int] = []
    for column in ("parent", "child"):
        if column in edges.columns:
            values.extend(layer for layer in edges[column].map(node_layer) if layer is not None and layer >= 0)
    if "node" in nodes.columns:
        values.extend(layer for layer in nodes["node"].map(node_layer) if layer is not None and layer >= 0)
    return max(values) if values else 0


def layer_label(layer: int, max_layer: int) -> str:
    if layer < 0:
        return "input"
    if layer > max_layer:
        return "logits"
    return f"L{layer}"


def save_budget_curve(budget: pd.DataFrame, out_path: Path) -> None:
    frame = budget.copy()
    if "budget_fraction" not in frame.columns:
        max_budget = float(frame["collapsed_edge_budget"].max())
        frame["budget_fraction"] = frame["collapsed_edge_budget"] / max_budget if max_budget else 0.0
    frame = frame.sort_values("budget_fraction")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    x = frame["budget_fraction"]
    for mean_col, std_col, label, color in (
        ("faithfulness_mean", "faithfulness_std", "Faithfulness", "#2f6fbb"),
        ("accuracy_mean", "accuracy_std", "Accuracy", "#2f8f5b"),
    ):
        y = frame[mean_col]
        ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=label, color=color)
        if std_col in frame:
            lower = (y - frame[std_col]).clip(lower=0)
            upper = (y + frame[std_col]).clip(upper=1)
            ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(RELATIVE_BUDGET_TICKVALS))
    ax.xaxis.set_major_formatter(FixedFormatter(RELATIVE_BUDGET_TICKTEXT))
    ax.set_xlabel("Collapsed edge budget (% of ranked model edges)")
    ax.set_ylabel("Mean score")
    ax.set_title("EAP Budget Sweep by Relative Edge Budget")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_attention_scores(nodes: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, row in nodes.iterrows():
        match = ATTN_RE.match(str(row["node"]))
        if match:
            rows.append(
                {
                    "layer": int(match.group(1)),
                    "head": int(match.group(2)),
                    "score": float(row["induced_score"]),
                }
            )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    heatmap = frame.pivot_table(index="layer", columns="head", values="score", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    sns.heatmap(heatmap, cmap="viridis", ax=ax, cbar_kws={"label": "Induced score"})
    ax.set_title("Attention Head Induced Scores")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_mlp_scores(nodes: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, row in nodes.iterrows():
        match = MLP_RE.match(str(row["node"]))
        if match:
            rows.append({"layer": int(match.group(1)), "score": float(row["induced_score"])})
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values("layer")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.bar(frame["layer"].astype(str), frame["score"], color="#4269a8")
    ax.set_title("MLP Layer Induced Scores")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Induced score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_layer_flow(edges: pd.DataFrame, nodes: pd.DataFrame, out_path: Path) -> None:
    max_layer = max_model_layer(edges, nodes)
    frame = edges.copy()
    frame["parent_layer"] = frame["parent"].map(lambda node: node_layer(node, max_layer))
    frame["child_layer"] = frame["child"].map(lambda node: node_layer(node, max_layer))
    frame = frame.dropna(subset=["parent_layer", "child_layer"])
    frame = frame[
        (frame["parent_layer"] >= 0)
        & (frame["parent_layer"] <= max_layer)
        & (frame["child_layer"] >= 0)
        & (frame["child_layer"] <= max_layer)
    ]
    if frame.empty:
        return
    frame["parent_layer"] = frame["parent_layer"].astype(int)
    frame["child_layer"] = frame["child_layer"].astype(int)
    grouped = (
        frame.groupby(["parent_layer", "child_layer"], as_index=False)["abs_score"]
        .sum()
        .sort_values(["parent_layer", "child_layer"])
    )
    layers = list(range(0, max_layer + 1))
    heatmap = grouped.pivot_table(
        index="parent_layer",
        columns="child_layer",
        values="abs_score",
        fill_value=0.0,
    ).reindex(index=layers, columns=layers, fill_value=0.0)
    labels = [layer_label(layer, max_layer) for layer in layers]
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    sns.heatmap(
        heatmap,
        cmap="Blues",
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Summed |edge score|"},
    )
    ax.set_title("Layer-to-Layer EAP Edge Mass")
    ax.set_xlabel("Child layer")
    ax.set_ylabel("Parent layer")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_layered_circuit(edges: pd.DataFrame, nodes: pd.DataFrame, out_path: Path, top_k: int) -> None:
    top_edges = edges.sort_values("abs_score", ascending=False).head(top_k).copy()
    if top_edges.empty:
        return
    max_layer = max_model_layer(top_edges, nodes)
    graph = nx.DiGraph()
    for _, row in top_edges.iterrows():
        graph.add_edge(str(row["parent"]), str(row["child"]), weight=float(row["abs_score"]))

    by_layer: dict[int, list[str]] = {}
    for node in graph.nodes:
        layer = node_layer(node, max_layer)
        if layer is None:
            continue
        by_layer.setdefault(layer, []).append(node)

    pos = {}
    for layer, layer_nodes in sorted(by_layer.items()):
        layer_nodes = sorted(layer_nodes)
        count = len(layer_nodes)
        for index, node in enumerate(layer_nodes):
            y = 0.0 if count == 1 else (count - 1) / 2 - index
            pos[node] = (layer, y)

    weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)], dtype=float)
    max_weight = float(weights.max()) if len(weights) else 1.0
    edge_widths = [1.0 + 5.0 * math.sqrt(data["weight"] / max_weight) for _, _, data in graph.edges(data=True)]
    node_colors = []
    for node in graph.nodes:
        kind = node_kind(node)
        if kind == "attention":
            node_colors.append("#4c78a8")
        elif kind == "mlp":
            node_colors.append("#f58518")
        elif kind == "input":
            node_colors.append("#54a24b")
        else:
            node_colors.append("#b279a2")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=14,
        width=edge_widths,
        edge_color="#666666",
        alpha=0.65,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=900,
        linewidths=1.0,
        edgecolors="white",
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    for layer in range(-1, max_layer + 2):
        ax.axvline(layer, color="#dddddd", linewidth=0.8, zorder=0)
        ax.text(layer, ax.get_ylim()[1], layer_label(layer, max_layer), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Top {len(top_edges)} EAP Edges")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = resolve_animacy_circuit_root(args.start_path)
    model_name = canonical_model_name(args.model)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(project_root, model_name, args.day, args.pipeline, args.dataset_set)
    )
    stem = PIPELINE_STEMS[args.pipeline]
    image_dir = resolve_image_output_dir(project_root, output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    edges = require_csv(output_dir / f"{stem}_edges_{args.day}.csv")
    nodes = require_csv(output_dir / f"{stem}_nodes_{args.day}.csv")
    budget = require_csv(output_dir / f"{stem}_budget_sweep_{args.day}.csv")

    artifacts: dict[str, str] = {}
    targets = {
        f"layered_circuit_top_{args.top_k_edges}_edges_{args.day}": lambda path: save_layered_circuit(edges, nodes, path, args.top_k_edges),
        f"budget_sweep_curve_{args.day}": lambda path: save_budget_curve(budget, path),
        f"attention_head_induced_scores_{args.day}": lambda path: save_attention_scores(nodes, path),
        f"mlp_layer_induced_scores_{args.day}": lambda path: save_mlp_scores(nodes, path),
        f"layer_flow_abs_scores_{args.day}": lambda path: save_layer_flow(edges, nodes, path),
    }
    for name, writer in targets.items():
        path = image_dir / f"{name}.png"
        writer(path)
        if path.exists():
            artifacts[name] = str(path)
            print(f"Wrote {path}")

    manifest_path = image_dir / f"{stem}_static_pngs_{args.day}.json"
    save_json(manifest_path, artifacts)
    print(f"Wrote PNG manifest: {manifest_path}")


if __name__ == "__main__":
    main()

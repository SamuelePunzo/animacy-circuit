from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from circuit_finder_core import (
    build_budget_circuit,
    build_graph,
    canonical_model_name,
    date_tag,
    ensure_dir,
    load_saved_ranked_edges,
    make_dataloader,
    make_eap_metrics,
    prepare_filtered_model_inputs,
    resolve_animacy_circuit_root,
    safe_model_name,
)


def bytes_to_gib(value: int) -> float:
    return float(value) / (1024**3)


def current_cuda_memory() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "free_gib": None,
            "total_gib": None,
            "allocated_gib": None,
            "reserved_gib": None,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_gib": bytes_to_gib(free_bytes),
        "total_gib": bytes_to_gib(total_bytes),
        "allocated_gib": bytes_to_gib(torch.cuda.memory_allocated()),
        "reserved_gib": bytes_to_gib(torch.cuda.memory_reserved()),
    }


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


def select_valid_target_pool(prepared: dict[str, Any], required_examples: int) -> tuple[str, pd.DataFrame]:
    # Strongest choice: examples already valid for the target model's normalized-recovery metric.
    target_raw = prepared["target_raw_scored_df"].copy()
    if {"clean_metric", "corrupt_metric"}.issubset(target_raw.columns):
        margin = target_raw["clean_metric"] - target_raw["corrupt_metric"]
        valid_target_raw = target_raw.loc[margin > 1e-6].reset_index(drop=True)
        if len(valid_target_raw) >= required_examples:
            return "target_raw_scored_df_margin_valid", valid_target_raw

    filtered_df = prepared["filtered_df"].reset_index(drop=True).copy()
    if len(filtered_df) >= required_examples:
        return "filtered_df", filtered_df

    target_scored_df = prepared["target_scored_df"].copy()
    if {"clean_metric", "corrupt_metric"}.issubset(target_scored_df.columns):
        margin = target_scored_df["clean_metric"] - target_scored_df["corrupt_metric"]
        valid_target_scored = target_scored_df.loc[margin > 1e-6].reset_index(drop=True)
        if len(valid_target_scored) >= required_examples:
            return "target_scored_df_margin_valid", valid_target_scored
        if not valid_target_scored.empty:
            return "target_scored_df_margin_valid_partial", valid_target_scored

    if not filtered_df.empty:
        return "filtered_df_partial", filtered_df

    if {"clean_metric", "corrupt_metric"}.issubset(target_raw.columns):
        margin = target_raw["clean_metric"] - target_raw["corrupt_metric"]
        valid_target_raw = target_raw.loc[margin > 1e-6].reset_index(drop=True)
        if not valid_target_raw.empty:
            return "target_raw_scored_df_margin_valid_partial", valid_target_raw

    source_success_df = prepared["source_success_df"].reset_index(drop=True).copy()
    return "source_success_df_fallback", source_success_df


def run_probe(
    *,
    model,
    evaluation_pool_df: pd.DataFrame,
    candidate_graph,
    metrics: dict[str, Any],
    batch_size: int,
    output_path: Path,
) -> dict[str, Any]:
    probe_df = evaluation_pool_df.head(batch_size).reset_index(drop=True)
    row: dict[str, Any] = {
        "evaluation_batch_size": int(batch_size),
        "example_count": int(len(probe_df)),
        "batch_count": 1 if len(probe_df) == batch_size else 0,
        "status": "pending",
    }

    if len(probe_df) < batch_size:
        row["status"] = "skipped_insufficient_examples"
        output_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    before = current_cuda_memory()
    row.update({f"before_{key}": value for key, value in before.items()})

    validation_loader = make_dataloader(probe_df, batch_size=batch_size, shuffle=False)

    started = time.perf_counter()
    try:
        from eap.evaluate import evaluate_graph

        results = evaluate_graph(
            model,
            candidate_graph,
            validation_loader,
            [metrics["accuracy"], metrics["kl_to_clean"]],
            quiet=True,
            intervention="patching",
            skip_clean=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        accuracy, kl_to_clean = results
        row.update(
            {
                "status": "ok",
                "elapsed_sec": float(elapsed),
                "accuracy_mean": float(accuracy.float().mean().item()),
                "kl_clean_mean": float(kl_to_clean.float().mean().item()),
            }
        )
    except RuntimeError as exc:
        elapsed = time.perf_counter() - started
        message = str(exc)
        row.update(
            {
                "status": "oom" if "out of memory" in message.lower() else "runtime_error",
                "elapsed_sec": float(elapsed),
                "error": message,
            }
        )
    except Exception as exc:  # Keep the sweep going if one batch hits invalid examples or other probe-specific issues.
        elapsed = time.perf_counter() - started
        row.update(
            {
                "status": "error",
                "elapsed_sec": float(elapsed),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        after = current_cuda_memory()
        row.update({f"after_{key}": value for key, value in after.items()})
        if torch.cuda.is_available():
            row["peak_allocated_gib"] = bytes_to_gib(torch.cuda.max_memory_allocated())
            row["peak_reserved_gib"] = bytes_to_gib(torch.cuda.max_memory_reserved())
            torch.cuda.empty_cache()
        output_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Llama evaluation batch sizes on A100.")
    parser.add_argument("--model", default="Llama 3.2 3B")
    parser.add_argument("--dataset-filter-model", default="gpt2")
    parser.add_argument("--filter-batch-size", type=int, default=4)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--max-filter-examples", type=int, default=512)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-day", default=None)
    parser.add_argument(
        "--edge-rankings-path",
        default=(
            "animacy-circuit/results/eap_ig/meta-llama_Llama-3.2-3B/"
            "model_specific_correct/2026-05-31/full_model/full_model_edges_2026-05-31.csv"
        ),
    )
    parser.add_argument(
        "--node-rankings-path",
        default=(
            "animacy-circuit/results/eap_ig/meta-llama_Llama-3.2-3B/"
            "model_specific_correct/2026-05-31/full_model/full_model_nodes_2026-05-31.csv"
        ),
    )
    parser.add_argument("--start-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_animacy_circuit_root(args.start_path)
    day = args.output_day or date_tag()
    model_name = canonical_model_name(args.model)
    output_root = ensure_dir(
        project_root
        / "results"
        / "smoke"
        / "eval_batch_size"
        / safe_model_name(model_name)
        / day
    )

    prepared = prepare_filtered_model_inputs(
        project_root=project_root,
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        metric_batch_size=args.filter_batch_size,
        seed=args.seed,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy="model_success",
    )
    required_examples = max(int(value) for value in args.batch_sizes)
    selected_pool_name, evaluation_pool_df = select_valid_target_pool(prepared, required_examples)

    sample_signature = {
        "kind": "evaluation_pool_only",
        "pool_name": selected_pool_name,
        "rows": int(len(evaluation_pool_df)),
        "seed": int(args.seed),
    }

    loaded = load_saved_ranked_edges(Path(args.edge_rankings_path), Path(args.node_rankings_path))
    if loaded is None:
        raise FileNotFoundError(f"Could not load ranked edges from {args.edge_rankings_path}")
    ranked_edges, edge_frame, node_frame = loaded

    budget = min(int(args.budget), len(ranked_edges))
    scored_graph = build_graph(prepared["model"])
    candidate_graph = build_budget_circuit(scored_graph, ranked_edges, budget)
    metrics = make_eap_metrics(prepared["animate_ids_tensor"], prepared["inanimate_ids_tensor"])
    metrics["kl_to_clean"] = make_kl_to_clean_metric()

    rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        probe_path = output_root / f"probe_eval_bs_{int(batch_size)}.json"
        row = run_probe(
            model=prepared["model"],
            evaluation_pool_df=evaluation_pool_df,
            candidate_graph=candidate_graph,
            metrics=metrics,
            batch_size=int(batch_size),
            output_path=probe_path,
        )
        rows.append(row)
        print(
            f"eval_bs={batch_size} status={row['status']} "
            f"peak_reserved_gib={row.get('peak_reserved_gib')} elapsed_sec={row.get('elapsed_sec')}"
        )

    summary = pd.DataFrame(rows)
    summary_path = output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "experiment": "llama_eval_batch_smoke",
        "model_name": model_name,
        "dataset_filter_model_name": canonical_model_name(args.dataset_filter_model),
        "filter_batch_size": int(args.filter_batch_size),
        "batch_sizes": [int(value) for value in args.batch_sizes],
        "budget": int(budget),
        "max_filter_examples": int(args.max_filter_examples),
        "seed": int(args.seed),
        "sample_signature": sample_signature,
        "evaluation_pool_count": int(len(evaluation_pool_df)),
        "edge_rankings_path": str(Path(args.edge_rankings_path)),
        "node_rankings_path": str(Path(args.node_rankings_path)),
        "edge_count": int(len(edge_frame)),
        "node_count": int(len(node_frame)),
        "paths": {
            "output_root": str(output_root),
            "summary_csv": str(summary_path),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved smoke results to {output_root}")


if __name__ == "__main__":
    main()

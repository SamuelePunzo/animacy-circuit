from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circuit_finder_core import (  # noqa: E402
    MODEL_SPECIFIC_CORRECT,
    CircuitPairDataset,
    add_sequence_lengths,
    build_budget_circuit,
    build_graph,
    canonical_model_name,
    compute_sequence_metrics,
    first_budget_reaching_faithfulness,
    load_model,
    make_eap_metrics,
    parse_ranked_edge_frame,
    resolve_animacy_circuit_root,
    resolve_target_source_path,
    resolve_shadow_source_artifacts,
    safe_model_name,
    sample_discovery_validation,
    save_csv,
    save_json,
    target_source_slug,
    tokenization_filter_jsonl_pairs_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved 85%-faithfulness EAP circuit on the named-entity "
            "target task using prefixes truncated from 'by the' to 'by'."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset-set", default=MODEL_SPECIFIC_CORRECT)
    parser.add_argument(
        "--target-source",
        default="dataset/semantic_meaningful/named_entity_targets.json",
        help="Target JSON path or target-source alias.",
    )
    parser.add_argument(
        "--main-experiment-path",
        default=None,
        help="Saved full_model directory or artifact path. Defaults to latest matching run.",
    )
    parser.add_argument("--source-faithfulness-threshold", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--discovery-sample-size", type=int, default=500)
    parser.add_argument("--discovery-margin-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--refresh-target-filter", action="store_true")
    parser.add_argument(
        "--target-token-mode",
        choices=("first_token", "whole_entity_single_token"),
        default="first_token",
        help=(
            "For truncated 'by' prefixes, first_token evaluates the first token "
            "of each named entity. whole_entity_single_token keeps only entities "
            "that are a single tokenizer token."
        ),
    )
    parser.add_argument("--output-day", default=None)
    return parser.parse_args()


def strip_terminal_article(prefix: str) -> str:
    suffix = " by the"
    if not prefix.endswith(suffix):
        raise ValueError(f"Expected prefix to end with {suffix!r}: {prefix!r}")
    return prefix[: -len(" the")]


def load_truncated_pairs(project_root: Path, model_name: str, max_examples: int | None) -> pd.DataFrame:
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
            row["clean_prefix"] = strip_terminal_article(str(clean))
            row["corrupt_prefix"] = strip_terminal_article(str(corrupt))
            rows.append(row)
            if max_examples is not None and len(rows) >= max_examples:
                break

    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return pd.DataFrame(rows).drop_duplicates(subset=["clean_prefix", "corrupt_prefix"]).reset_index(drop=True)


def filter_model_success(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[
        (scored["clean_metric"] > 0.0)
        & (scored["corrupt_metric"] < 0.0)
        & ((scored["clean_metric"] - scored["corrupt_metric"]) > 1e-6)
    ].reset_index(drop=True)


def load_raw_target_sets(project_root: Path, target_source: str) -> tuple[list[str], list[str], Path]:
    path = resolve_target_source_path(project_root, target_source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload["targets"]
    return list(targets["animate"]), list(targets["inanimate"]), path


def first_token_targets(
    entities: list[str],
    tokenizer,
    device: str | torch.device,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    target_ids: list[int] = []
    for entity in entities:
        token_ids = tokenizer(" " + entity, add_special_tokens=False).input_ids
        if not token_ids:
            rows.append(
                {
                    "entity": entity,
                    "status": "dropped_empty_tokenization",
                    "token_ids": [],
                }
            )
            continue
        first_id = int(token_ids[0])
        first_token = tokenizer.decode([first_id]).strip()
        is_duplicate = first_id in seen_ids
        rows.append(
            {
                "entity": entity,
                "first_token": first_token,
                "first_token_id": first_id,
                "token_ids": [int(token_id) for token_id in token_ids],
                "token_count": len(token_ids),
                "status": "duplicate_first_token" if is_duplicate else "kept",
            }
        )
        if not is_duplicate:
            seen_ids.add(first_id)
            target_ids.append(first_id)
    if not target_ids:
        raise ValueError("No first-token target IDs were produced.")
    return torch.tensor(target_ids, dtype=torch.long, device=device), rows


def whole_entity_single_token_targets(
    entities: list[str],
    tokenizer,
    device: str | torch.device,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    target_ids: list[int] = []
    for entity in entities:
        token_ids = tokenizer(" " + entity, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            rows.append(
                {
                    "entity": entity,
                    "status": "dropped_not_single_token",
                    "token_ids": [int(token_id) for token_id in token_ids],
                    "token_count": len(token_ids),
                }
            )
            continue
        token_id = int(token_ids[0])
        is_duplicate = token_id in seen_ids
        rows.append(
            {
                "entity": entity,
                "target_token": tokenizer.decode([token_id]).strip(),
                "target_token_id": token_id,
                "token_ids": [token_id],
                "token_count": 1,
                "status": "duplicate_target_token" if is_duplicate else "kept",
            }
        )
        if not is_duplicate:
            seen_ids.add(token_id)
            target_ids.append(token_id)
    if not target_ids:
        raise ValueError("No whole-entity single-token target IDs were produced.")
    return torch.tensor(target_ids, dtype=torch.long, device=device), rows


def named_entity_target_tensors(
    project_root: Path,
    target_source: str,
    tokenizer,
    device: str | torch.device,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object], Path]:
    animate_entities, inanimate_entities, target_path = load_raw_target_sets(project_root, target_source)
    if mode == "first_token":
        animate_ids, animate_rows = first_token_targets(animate_entities, tokenizer, device)
        inanimate_ids, inanimate_rows = first_token_targets(inanimate_entities, tokenizer, device)
    elif mode == "whole_entity_single_token":
        animate_ids, animate_rows = whole_entity_single_token_targets(animate_entities, tokenizer, device)
        inanimate_ids, inanimate_rows = whole_entity_single_token_targets(inanimate_entities, tokenizer, device)
    else:
        raise ValueError(f"Unsupported target token mode: {mode}")

    def count_status(rows: list[dict[str, object]], status: str) -> int:
        return sum(1 for row in rows if row.get("status") == status)

    summary = {
        "mode": mode,
        "animate": {
            "entity_count": len(animate_entities),
            "target_token_count": int(animate_ids.numel()),
            "duplicate_count": count_status(animate_rows, "duplicate_first_token")
            + count_status(animate_rows, "duplicate_target_token"),
            "kept_examples": [row for row in animate_rows if row.get("status") == "kept"][:20],
            "dropped_examples": [row for row in animate_rows if str(row.get("status", "")).startswith("dropped")][:20],
        },
        "inanimate": {
            "entity_count": len(inanimate_entities),
            "target_token_count": int(inanimate_ids.numel()),
            "duplicate_count": count_status(inanimate_rows, "duplicate_first_token")
            + count_status(inanimate_rows, "duplicate_target_token"),
            "kept_examples": [row for row in inanimate_rows if row.get("status") == "kept"][:20],
            "dropped_examples": [row for row in inanimate_rows if str(row.get("status", "")).startswith("dropped")][:20],
        },
    }
    return animate_ids, inanimate_ids, summary, target_path


def resolve_valid_source_artifacts(
    project_root: Path,
    model_name: str,
    dataset_set: str,
    main_experiment_path: str | None,
) -> dict[str, Path | None]:
    if main_experiment_path is not None:
        return resolve_shadow_source_artifacts(
            project_root,
            model_name,
            dataset_set,
            main_experiment_path=main_experiment_path,
        )

    model_slug = safe_model_name(model_name)
    base = project_root / "results" / "eap_ig" / model_slug / dataset_set
    candidates = sorted(
        base.glob("*/full_model"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            return resolve_shadow_source_artifacts(
                project_root,
                model_name,
                dataset_set,
                main_experiment_path=candidate,
            )
        except FileNotFoundError as error:
            errors.append(str(error))

    if errors:
        raise FileNotFoundError(errors[0])
    raise FileNotFoundError(f"No full_model source directories found under {base}")


def main() -> None:
    args = parse_args()
    project_root = resolve_animacy_circuit_root(Path.cwd())
    model_name = canonical_model_name(args.model)
    model_slug = safe_model_name(model_name)
    target_slug = target_source_slug(project_root, args.target_source)
    day = args.output_day or f"named_entity_circuit_eval_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}"
    output_dir = (
        project_root
        / "results"
        / "named_entity_circuit_eval"
        / model_slug
        / args.dataset_set
        / day
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = resolve_valid_source_artifacts(
        project_root,
        model_name,
        args.dataset_set,
        args.main_experiment_path,
    )
    edge_frame = pd.read_csv(source_paths["edge_path"])
    budget_frame = pd.read_csv(source_paths["budget_path"])
    source_ranked_edges = parse_ranked_edge_frame(edge_frame)
    threshold_row = first_budget_reaching_faithfulness(
        budget_frame,
        args.source_faithfulness_threshold,
    )
    circuit_budget = int(threshold_row["collapsed_edge_budget"])

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

    raw_df = add_sequence_lengths(
        load_truncated_pairs(project_root, model_name, args.max_examples),
        model,
    )
    scored_df = compute_sequence_metrics(
        raw_df,
        model,
        tokenizer,
        animate_ids,
        inanimate_ids,
        batch_size=args.batch_size,
    )
    model_success_df = filter_model_success(scored_df)

    scored_path = output_dir / f"named_entity_truncated_scored_{day}.csv"
    success_path = output_dir / f"named_entity_truncated_model_success_{day}.csv"
    save_csv(scored_df, scored_path, index=False)
    save_csv(model_success_df, success_path, index=False)

    discovery_df, validation_df, sample_signature = sample_discovery_validation(
        model_success_df,
        discovery_sample_size=args.discovery_sample_size,
        seed=args.seed,
        discovery_margin_threshold=args.discovery_margin_threshold,
    )

    scored_graph = build_graph(model)
    circuit_graph = build_budget_circuit(scored_graph, source_ranked_edges, circuit_budget)
    validation_loader = DataLoader(
        CircuitPairDataset(validation_df),
        batch_size=args.evaluation_batch_size,
        shuffle=False,
    )
    metrics = make_eap_metrics(animate_ids, inanimate_ids)
    from eap.evaluate import evaluate_graph

    with torch.no_grad():
        faithfulness_values, accuracy_values = evaluate_graph(
            model,
            circuit_graph,
            validation_loader,
            [metrics["faithfulness"], metrics["accuracy"]],
            quiet=True,
            intervention="patching",
            skip_clean=False,
        )

    eval_path = output_dir / f"named_entity_85pct_circuit_eval_{day}.json"

    faithfulness_cpu = faithfulness_values.float().cpu()
    accuracy_cpu = accuracy_values.float().cpu()
    summary = {
        "experiment": "named_entity_85pct_circuit_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset_set": args.dataset_set,
        "target_source": str(args.target_source),
        "target_source_slug": target_slug,
        "target_token_mode": args.target_token_mode,
        "target_source_path": str(target_path),
        "source_full_model_dir": str(source_paths["source_dir"]),
        "source_edge_path": str(source_paths["edge_path"]),
        "source_budget_path": str(source_paths["budget_path"]),
        "source_threshold_row": threshold_row,
        "evaluated_circuit_budget": circuit_budget,
        "target_counts": {
            "animate": int(animate_ids.numel()),
            "inanimate": int(inanimate_ids.numel()),
        },
        "target_filter_summary": target_summary,
        "dataset_counts": {
            "truncated_scored": int(len(scored_df)),
            "model_success": int(len(model_success_df)),
            "discovery": int(len(discovery_df)),
            "validation": int(len(validation_df)),
            "discovery_sample_signature": sample_signature,
            "discovery_margin_threshold": args.discovery_margin_threshold,
        },
        "baseline_metrics": {
            "clean_metric_mean": float(scored_df["clean_metric"].mean()),
            "corrupt_metric_mean": float(scored_df["corrupt_metric"].mean()),
            "model_success_clean_metric_mean": float(model_success_df["clean_metric"].mean()),
            "model_success_corrupt_metric_mean": float(model_success_df["corrupt_metric"].mean()),
        },
        "circuit_eval": {
            "faithfulness_mean": float(faithfulness_cpu.mean().item()),
            "faithfulness_std": (
                float(faithfulness_cpu.std(unbiased=False).item())
                if len(faithfulness_cpu) > 1
                else 0.0
            ),
            "accuracy_mean": float(accuracy_cpu.mean().item()),
            "accuracy_std": (
                float(accuracy_cpu.std(unbiased=False).item())
                if len(accuracy_cpu) > 1
                else 0.0
            ),
            "validation_examples": int(len(faithfulness_cpu)),
            "expanded_edge_count": int(circuit_graph.count_included_edges()),
            "induced_node_count": int(circuit_graph.count_included_nodes() - 2),
        },
        "paths": {
            "output_dir": str(output_dir),
            "scored_dataset": str(scored_path),
            "model_success_dataset": str(success_path),
            "summary": str(eval_path),
        },
    }
    save_json(eval_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

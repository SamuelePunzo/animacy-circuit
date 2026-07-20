from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from circuit_finder_core import (
    add_concept_verb_positions,
    add_sequence_lengths,
    generate_exact_length_batches,
    load_model,
    normalize_concept_pair_metadata,
    resolve_animacy_circuit_root,
)
from run_concept_analysis_artifacts import latest_file, resolve_run_dir
from run_necessary_semantic_activations import model_name_from_slug
from utils import save_csv, save_json, timestamp_tag

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


HOOK_POINT_ORDER = ("pre", "mid", "post")


def residual_hook_name(layer: int, hook_point: str) -> str:
    return f"blocks.{layer}.hook_resid_{hook_point}"


def residual_hooks(model) -> list[dict[str, Any]]:
    available = set(model.hook_dict.keys()) if hasattr(model, "hook_dict") else set()
    rows: list[dict[str, Any]] = []
    for layer in range(int(model.cfg.n_layers)):
        for hook_point in HOOK_POINT_ORDER:
            hook_name = residual_hook_name(layer, hook_point)
            if not available or hook_name in available:
                rows.append({"layer": layer, "hook_point": hook_point, "hook_name": hook_name})
    return rows


def chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def compute_residual_stream_pca(
    *,
    run_dir: Path,
    model_name: str,
    batch_size: int,
    hook_chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if PCA is None:
        raise RuntimeError("scikit-learn is not available; cannot run PCA.")

    split_path = latest_file(run_dir, "test_split_*.csv", required=True)
    assert split_path is not None
    test_split = pd.read_csv(split_path)
    test_df = normalize_concept_pair_metadata(test_split)

    model = load_model(model_name)
    test_df = add_sequence_lengths(test_df, model)
    test_df = add_concept_verb_positions(test_df, model.tokenizer)

    hooks = residual_hooks(model)
    if not hooks:
        raise ValueError(f"No residual stream hooks found for model {model_name!r}.")

    hook_names = [row["hook_name"] for row in hooks]
    hook_meta = {row["hook_name"]: row for row in hooks}
    clean_acts: dict[str, list[torch.Tensor]] = {hook_name: [] for hook_name in hook_names}
    corrupt_acts: dict[str, list[torch.Tensor]] = {hook_name: [] for hook_name in hook_names}
    records: list[dict[str, Any]] = []
    hook_chunks = chunked(hooks, max(int(hook_chunk_size), 1))

    for hook_chunk_index, hook_chunk in enumerate(tqdm(hook_chunks, desc="hook chunks")):
        chunk_hook_names = [row["hook_name"] for row in hook_chunk]
        batches = generate_exact_length_batches(test_df, model, batch_size, model.cfg.device)
        for clean_tokens, corrupt_tokens, batch_df in tqdm(
            batches,
            desc=f"{chunk_hook_names[0]}..",
            leave=False,
        ):
            positions = torch.tensor(
                batch_df["verb_token_position"].to_numpy(dtype="int64"),
                dtype=torch.long,
                device=model.cfg.device,
            )
            batch_indices = torch.arange(clean_tokens.shape[0], device=model.cfg.device)
            with torch.no_grad():
                _, clean_cache = model.run_with_cache(clean_tokens, names_filter=chunk_hook_names)
                _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=chunk_hook_names)

            for hook_name in chunk_hook_names:
                clean_selected = clean_cache[hook_name][batch_indices, positions, :].detach().float().cpu()
                corrupt_selected = corrupt_cache[hook_name][batch_indices, positions, :].detach().float().cpu()
                clean_acts[hook_name].append(clean_selected)
                corrupt_acts[hook_name].append(corrupt_selected)
            if hook_chunk_index == 0:
                records.extend(batch_df.to_dict("records"))
            del clean_cache, corrupt_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    base_df = pd.DataFrame(records).reset_index().rename(columns={"index": "pair_id"})
    pca_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for hook_name in hook_names:
        clean_tensor = torch.cat(clean_acts[hook_name], dim=0)
        corrupt_tensor = torch.cat(corrupt_acts[hook_name], dim=0)
        combined = torch.cat([clean_tensor, corrupt_tensor], dim=0).numpy()
        coords = PCA(n_components=2, random_state=0).fit_transform(combined)
        meta = hook_meta[hook_name]
        n = len(base_df)

        clean_points = base_df.copy()
        clean_points["pc1"] = coords[:n, 0]
        clean_points["pc2"] = coords[:n, 1]
        clean_points["sentence_type"] = "clean"
        clean_points["active_verb"] = clean_points["clean_verb"]
        clean_points["active_sentence"] = clean_points["clean_prefix"]

        corrupt_points = base_df.copy()
        corrupt_points["pc1"] = coords[n:, 0]
        corrupt_points["pc2"] = coords[n:, 1]
        corrupt_points["sentence_type"] = "corrupt"
        corrupt_points["active_verb"] = corrupt_points["corrupt_verb"]
        corrupt_points["active_sentence"] = corrupt_points["corrupt_prefix"]

        hook_points = pd.concat([clean_points, corrupt_points], ignore_index=True)
        hook_points["layer"] = int(meta["layer"])
        hook_points["hook_point"] = str(meta["hook_point"])
        hook_points["hook_name"] = hook_name
        pca_frames.append(hook_points)

        summary_rows.append(
            {
                "layer": int(meta["layer"]),
                "hook_point": str(meta["hook_point"]),
                "hook_name": hook_name,
                "row_count": int(len(hook_points)),
            }
        )

    pca_points = pd.concat(pca_frames, ignore_index=True)
    summary = {
        "model_name": model_name,
        "run_dir": str(run_dir),
        "test_examples": int(len(base_df)),
        "hook_count": int(len(hooks)),
        "hook_chunk_size": int(hook_chunk_size),
        "batch_size": int(batch_size),
        "hooks": summary_rows,
    }
    return pca_points, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute PCA of verb-token residual stream activations across layers.")
    parser.add_argument("--model", default="gpt2", help="Model slug under results/concept_extraction.")
    parser.add_argument("--day", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--start-path", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hook-chunk-size", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_animacy_circuit_root(args.start_path)
    run_dir = resolve_run_dir(project_root=project_root, model_slug=args.model, day=args.day, run_dir=args.run_dir)
    summary_path = latest_file(run_dir, "concept_extraction_summary_*.json", required=False)
    if summary_path is not None:
        concept_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = str(concept_summary.get("model_name", model_name_from_slug(str(args.model))))
    else:
        model_name = model_name_from_slug(str(args.model))

    pca_points, summary = compute_residual_stream_pca(
        run_dir=run_dir,
        model_name=model_name,
        batch_size=args.batch_size,
        hook_chunk_size=args.hook_chunk_size,
    )
    tag = timestamp_tag()
    pca_path = run_dir / f"residual_stream_pca_points_{tag}.csv"
    summary_path = run_dir / f"residual_stream_pca_summary_{tag}.json"
    save_csv(pca_points, pca_path, index=False)
    save_json(summary_path, summary)
    print(f"Wrote residual stream PCA points: {pca_path}")
    print(f"Wrote residual stream PCA summary: {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from circuit_finder_core import generate_exact_length_batches, resolve_animacy_circuit_root
from run_necessary_circuit_diagnostics import (
    accumulate_attention_patterns,
    attention_summary_from_accumulators,
    component_sort_key,
    empty_attention_accumulators,
    positions_for_condition,
)
from run_necessary_semantic_activations import (
    SLOT_COLUMNS,
    add_key_positions,
    batch_size_for_model,
    build_prompt_intersection,
    component_hook_spec,
    load_report,
    load_slot_examples,
    sorted_slots,
)


def latest_report_dir(results_root: Path) -> Path:
    preferred = sorted(results_root.glob("necessary_edge_expansion_main_original_20_50_*"))
    fallback = sorted(results_root.glob("necessary_edge_expansion_*"))
    candidates = preferred or fallback
    if not candidates:
        raise FileNotFoundError(f"No necessary-edge reports found under {results_root}")
    return candidates[-1]


def analyze_slot_attention(
    project_root: Path,
    component_cards: pd.DataFrame,
    slot_row: pd.Series,
    n_examples: int,
    batch_size: int,
    prompt_pairs: pd.DataFrame | None,
) -> pd.DataFrame:
    slot_components = component_cards.merge(pd.DataFrame([slot_row[SLOT_COLUMNS].to_dict()]), on=SLOT_COLUMNS, how="inner")
    components = sorted(slot_components["component"].dropna().astype(str).drop_duplicates().tolist(), key=component_sort_key)
    component_specs = {component: component_hook_spec(component) for component in components}
    heads = [
        component
        for component, spec in component_specs.items()
        if spec.get("component_type") == "attn" and spec.get("pattern_hook")
    ]
    if not heads:
        return pd.DataFrame()

    examples, context = load_slot_examples(project_root, slot_row, n_examples, batch_size, prompt_pairs=prompt_pairs)
    model = context["model"]
    tokenizer = context["tokenizer"]
    examples = add_key_positions(examples, tokenizer)

    head_specs = {component: component_specs[component] for component in heads}
    requested_hooks = sorted({spec["pattern_hook"] for spec in head_specs.values() if spec.get("pattern_hook")})
    available_hooks = set(name for name, _hook in model.hook_dict.items()) if hasattr(model, "hook_dict") else set(requested_hooks)
    hook_names = [hook for hook in requested_hooks if hook in available_hooks]
    unsupported_hooks = sorted(set(requested_hooks) - set(hook_names))
    if unsupported_hooks:
        print(f"Skipping unsupported attention hooks for {slot_row['model_slug']} seed {slot_row['seed']}: {unsupported_hooks}")

    attention_sums, attention_counts = empty_attention_accumulators(heads)
    estimated_batches = sum((len(group) + batch_size - 1) // batch_size for _, group in examples.groupby("seq_len"))
    batches = generate_exact_length_batches(examples, model, batch_size=batch_size, device=model.cfg.device)

    for clean_tokens, corrupt_tokens, batch_df in tqdm(
        batches,
        total=estimated_batches,
        desc=f"{slot_row['model_slug']} attention patterns",
    ):
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_names)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=hook_names)

        for batch_index, (_example_id, row) in enumerate(batch_df.iterrows()):
            accumulate_attention_patterns(
                attention_sums,
                attention_counts,
                head_specs,
                heads,
                "clean",
                clean_cache,
                batch_index,
                positions_for_condition(row, "clean"),
            )
            accumulate_attention_patterns(
                attention_sums,
                attention_counts,
                head_specs,
                heads,
                "corrupt",
                corrupt_cache,
                batch_index,
                positions_for_condition(row, "corrupt"),
            )
        del clean_cache, corrupt_cache

    del context["model"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return attention_summary_from_accumulators(slot_row, heads, attention_sums, attention_counts)


def write_outputs(output_dir: Path, attention_rows: pd.DataFrame, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not attention_rows.empty:
        attention_rows.to_csv(output_dir / "attention_pattern_summary.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute necessary-circuit attention-pattern summaries only.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
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
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = args.output_dir or (results_root / f"necessary_attention_patterns_{stamp}")
    seeds = None if args.all_seeds else args.seeds

    selected_summary, _semantic_edges, component_cards = load_report(report_dir, args.model_slug, seeds)
    prompt_pairs = None if args.no_prompt_intersection else build_prompt_intersection(project_root, selected_summary, args.batch_size)

    frames: list[pd.DataFrame] = []
    completed_slots: list[str] = []
    slot_batch_sizes: dict[str, int] = {}
    base_metadata = {
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "n_examples": int(args.n_examples),
        "batch_size_override": int(args.batch_size) if args.batch_size is not None else None,
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
            f"Analyzing attention patterns for {slot_row['model_slug']} / {slot_row['run_name']} / seed {slot_row['seed']} "
            f"with n_examples={args.n_examples}, batch_size={slot_batch_size}"
        )
        frame = analyze_slot_attention(
            project_root=project_root,
            component_cards=component_cards,
            slot_row=slot_row,
            n_examples=args.n_examples,
            batch_size=slot_batch_size,
            prompt_pairs=prompt_pairs,
        )
        if not frame.empty:
            frames.append(frame)
        completed_slots.append(slot_key)
        attention_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        metadata = {
            **base_metadata,
            "slot_batch_sizes": slot_batch_sizes,
            "completed_slots": completed_slots,
            "completed_slot_count": int(len(completed_slots)),
            "attention_pattern_rows": int(len(attention_rows)),
        }
        write_outputs(output_dir, attention_rows, metadata)
        print(f"Checkpointed attention patterns to {output_dir}")

    print(f"Wrote necessary-circuit attention patterns to {output_dir}")


if __name__ == "__main__":
    main()

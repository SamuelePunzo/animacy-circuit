from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from circuit_finder_core import (
    generate_exact_length_batches,
    is_single_token_target,
    load_animacy_targets,
    resolve_animacy_circuit_root,
    tokenizer_input_ids,
)
from run_necessary_semantic_activations import (
    SLOT_COLUMNS,
    add_key_positions,
    batch_size_for_model,
    build_prompt_intersection,
    component_hook_spec,
    hooks_for_components,
    load_report,
    load_slot_examples,
    sorted_slots,
)


SENTENCE_POSITION_ORDER = ["BOS", "The", "patient", "was", "verb", "by", "the"]


def component_sort_key(name: str) -> tuple[int, int, int, str]:
    spec = component_hook_spec(name)
    kind_order = {"attn": 0, "mlp": 1}
    return (
        kind_order.get(str(spec.get("component_type")), 9),
        int(spec.get("layer") if spec.get("layer") is not None else 999),
        int(spec.get("head") if spec.get("head") is not None else -1),
        str(name),
    )


def latest_report_dir(results_root: Path) -> Path:
    preferred = sorted(results_root.glob("necessary_edge_expansion_main_original_20_50_*"))
    fallback = sorted(results_root.glob("necessary_edge_expansion_*"))
    candidates = preferred or fallback
    if not candidates:
        raise FileNotFoundError(f"No necessary-edge reports found under {results_root}")
    return candidates[-1]


def condition_verbs(batch_df: pd.DataFrame, condition: str) -> list[str]:
    candidates = [f"{condition}_verb", f"{condition}_verb_x", f"{condition}_verb_y"]
    for column in candidates:
        if column in batch_df.columns:
            return batch_df[column].astype(str).tolist()
    raise KeyError(
        f"Could not find a verb column for condition={condition!r}. "
        f"Tried {candidates}; available columns are {sorted(batch_df.columns.tolist())}."
    )


def target_token_records(
    tokenizer,
    animate_words: list[str],
    inanimate_words: list[str],
    target_limit: int | None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target_set, words in [("animate", animate_words), ("inanimate", inanimate_words)]:
        kept = 0
        for word in words:
            if target_limit is not None and kept >= target_limit:
                break
            token_ids = tokenizer_input_ids(tokenizer, " " + str(word))
            if len(token_ids) != 1:
                continue
            token_id = int(token_ids[0])
            records.append(
                {
                    "target_set": target_set,
                    "target": str(word),
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id]),
                }
            )
            kept += 1
    rows = pd.DataFrame(records)
    if rows.empty:
        raise ValueError("No single-token target words available for logit-lens diagnostics.")
    return rows


def component_activation_at_final(cache, spec: dict[str, Any], final_positions: torch.Tensor) -> torch.Tensor:
    activation = cache[spec["activation_hook"]]
    batch_indices = torch.arange(final_positions.shape[0], device=activation.device)
    if spec["component_type"] == "attn":
        return activation[batch_indices, final_positions, int(spec["head"]), :]
    return activation[batch_indices, final_positions, :]


def tool_logit_lens_logits(model, component_vectors: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "ln_final") and model.ln_final is not None:
        vectors = model.ln_final(component_vectors)
    else:
        vectors = component_vectors
    if hasattr(model, "unembed") and callable(model.unembed):
        return model.unembed(vectors)
    if hasattr(model, "W_U"):
        return vectors @ model.W_U
    if hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        return vectors @ model.unembed.W_U
    raise AttributeError("Could not project component vectors to logits; model has no callable unembed or W_U.")


def empty_attention_accumulators(heads: list[str]) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    shape = (len(SENTENCE_POSITION_ORDER), len(SENTENCE_POSITION_ORDER))
    sums = {(condition, component): np.zeros(shape, dtype=float) for condition in ["clean", "corrupt"] for component in heads}
    counts = {(condition, component): np.zeros(shape, dtype=float) for condition in ["clean", "corrupt"] for component in heads}
    return sums, counts


def positions_for_condition(row: pd.Series, condition: str) -> list[Any]:
    prefix = "clean" if condition == "clean" else "corrupt"
    return [
        row["bos_pos"],
        row.get(f"{prefix}_initial_the_pos"),
        row.get(f"{prefix}_patient_pos"),
        row.get(f"{prefix}_was_pos"),
        row.get(f"{prefix}_verb_pos"),
        row.get(f"{prefix}_by_pos"),
        row["final_pos"],
    ]


def accumulate_attention_patterns(
    sums: dict[tuple[str, str], np.ndarray],
    counts: dict[tuple[str, str], np.ndarray],
    component_specs: dict[str, dict[str, Any]],
    heads: list[str],
    condition: str,
    cache,
    batch_index: int,
    positions: list[Any],
) -> None:
    for component in heads:
        spec = component_specs[component]
        pattern_hook = spec.get("pattern_hook")
        if not pattern_hook or pattern_hook not in cache:
            continue
        pattern = cache[pattern_hook][batch_index, int(spec["head"])].detach().float().cpu().numpy()
        for query_idx, query_pos in enumerate(positions):
            if query_pos is None or pd.isna(query_pos):
                continue
            query_pos = int(query_pos)
            if query_pos < 0 or query_pos >= pattern.shape[0]:
                continue
            for key_idx, key_pos in enumerate(positions):
                if key_pos is None or pd.isna(key_pos):
                    continue
                key_pos = int(key_pos)
                if key_pos < 0 or key_pos >= pattern.shape[1]:
                    continue
                sums[(condition, component)][query_idx, key_idx] += float(pattern[query_pos, key_pos])
                counts[(condition, component)][query_idx, key_idx] += 1.0


def attention_summary_from_accumulators(
    slot_row: pd.Series,
    heads: list[str],
    sums: dict[tuple[str, str], np.ndarray],
    counts: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = {col: slot_row[col] for col in SLOT_COLUMNS}
    base["selected_budget"] = int(slot_row["selected_budget"])
    for condition in ["clean", "corrupt"]:
        for component in heads:
            spec = component_hook_spec(component)
            matrix = np.divide(
                sums[(condition, component)],
                counts[(condition, component)],
                out=np.full_like(sums[(condition, component)], np.nan),
                where=counts[(condition, component)] > 0,
            )
            for query_idx, query_label in enumerate(SENTENCE_POSITION_ORDER):
                for key_idx, key_label in enumerate(SENTENCE_POSITION_ORDER):
                    rows.append(
                        {
                            **base,
                            "component": component,
                            "component_type": spec["component_type"],
                            "layer": spec["layer"],
                            "head": spec["head"],
                            "condition": condition,
                            "query_token": query_label,
                            "key_token": key_label,
                            "attention_mass_mean": float(matrix[query_idx, key_idx]),
                            "example_count": int(counts[(condition, component)][query_idx, key_idx]),
                        }
                    )
    return pd.DataFrame(rows)


def mlp_prompt_target_set_logit_lens_summary(target_logit_lens: pd.DataFrame) -> pd.DataFrame:
    if target_logit_lens.empty:
        return pd.DataFrame()
    mlp_rows = target_logit_lens[target_logit_lens["component_type"] == "mlp"].copy()
    if mlp_rows.empty:
        return pd.DataFrame()
    group_cols = SLOT_COLUMNS + [
        "selected_budget",
        "component",
        "component_type",
        "layer",
        "head",
        "condition",
        "target_set",
    ]
    summary = (
        mlp_rows.groupby(group_cols, dropna=False)
        .agg(
            average_target_set_logit=("mean_logit", "mean"),
            target_logit_std=("mean_logit", "std"),
            target_count=("target", "nunique"),
            target_token_count=("token_id", "nunique"),
            example_count=("example_count", "first"),
        )
        .reset_index()
    )
    pivot = (
        summary.pivot_table(
            index=SLOT_COLUMNS + ["selected_budget", "component", "component_type", "layer", "head", "condition"],
            columns="target_set",
            values="average_target_set_logit",
            aggfunc="first",
        )
        .reset_index()
    )
    if {"animate", "inanimate"}.issubset(pivot.columns):
        pivot["animate_minus_inanimate_logit"] = pivot["animate"] - pivot["inanimate"]
        summary = summary.merge(
            pivot[
                SLOT_COLUMNS
                + [
                    "selected_budget",
                    "component",
                    "component_type",
                    "layer",
                    "head",
                    "condition",
                    "animate_minus_inanimate_logit",
                ]
            ],
            on=SLOT_COLUMNS + ["selected_budget", "component", "component_type", "layer", "head", "condition"],
            how="left",
        )
    return summary.sort_values(["layer", "component", "condition", "target_set"], kind="stable").reset_index(drop=True)


def analyze_slot(
    project_root: Path,
    component_cards: pd.DataFrame,
    slot_row: pd.Series,
    n_examples: int,
    batch_size: int,
    target_limit: int | None,
    prompt_pairs: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    slot_components = component_cards.merge(pd.DataFrame([slot_row[SLOT_COLUMNS].to_dict()]), on=SLOT_COLUMNS, how="inner")
    components = sorted(slot_components["component"].dropna().astype(str).drop_duplicates().tolist(), key=component_sort_key)
    component_specs = {component: component_hook_spec(component) for component in components}
    component_specs = {component: spec for component, spec in component_specs.items() if spec.get("activation_hook")}
    components = list(component_specs.keys())
    heads = [component for component, spec in component_specs.items() if spec["component_type"] == "attn" and spec.get("pattern_hook")]
    if not components:
        return {}

    examples, context = load_slot_examples(project_root, slot_row, n_examples, batch_size, prompt_pairs=prompt_pairs)
    model = context["model"]
    tokenizer = context["tokenizer"]
    examples = add_key_positions(examples, tokenizer)

    animate_words, inanimate_words = load_animacy_targets(project_root, target_source=context.get("target_source"))
    animate_words = [word for word in animate_words if is_single_token_target(word, tokenizer)]
    inanimate_words = [word for word in inanimate_words if is_single_token_target(word, tokenizer)]
    target_df = target_token_records(tokenizer, animate_words, inanimate_words, target_limit)
    target_records = target_df.to_dict("records")
    target_ids = torch.tensor(target_df["token_id"].tolist(), device=model.cfg.device, dtype=torch.long)

    requested_hooks = hooks_for_components(components, include_patterns=True)
    available_hooks = set(name for name, _hook in model.hook_dict.items()) if hasattr(model, "hook_dict") else set(requested_hooks)
    hook_names = [hook for hook in requested_hooks if hook in available_hooks]
    unsupported_hooks = sorted(set(requested_hooks) - set(hook_names))
    if unsupported_hooks:
        print(f"Skipping unsupported hooks for {slot_row['model_slug']} seed {slot_row['seed']}: {unsupported_hooks}")

    token_logit_sums = {
        (condition, component): torch.zeros(len(target_df), dtype=torch.float64)
        for condition in ["clean", "corrupt"]
        for component in components
    }
    token_logit_counts = {(condition, component): 0 for condition in ["clean", "corrupt"] for component in components}
    verb_token_logit_sums: dict[tuple[str, str, str], torch.Tensor] = {}
    verb_token_logit_counts: dict[tuple[str, str, str], int] = {}
    attention_sums, attention_counts = empty_attention_accumulators(heads)

    estimated_batches = sum((len(group) + batch_size - 1) // batch_size for _, group in examples.groupby("seq_len"))
    batches = generate_exact_length_batches(examples, model, batch_size=batch_size, device=model.cfg.device)
    desc = f"{slot_row['model_slug']} diagnostics"
    for clean_tokens, corrupt_tokens, batch_df in tqdm(batches, total=estimated_batches, desc=desc):
        final_positions = torch.tensor((batch_df["seq_len"].astype(int) - 1).tolist(), device=model.cfg.device, dtype=torch.long)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_names)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=hook_names)

            for component, spec in component_specs.items():
                if spec["activation_hook"] not in clean_cache or spec["activation_hook"] not in corrupt_cache:
                    continue
                for condition, cache in [("clean", clean_cache), ("corrupt", corrupt_cache)]:
                    component_vectors = component_activation_at_final(cache, spec, final_positions)
                    logits = tool_logit_lens_logits(model, component_vectors.float())[:, target_ids]
                    token_logit_sums[(condition, component)] += logits.detach().double().sum(dim=0).cpu()
                    token_logit_counts[(condition, component)] += int(logits.shape[0])
                    logits_cpu = logits.detach().double().cpu()
                    for verb, row_logits in zip(condition_verbs(batch_df, condition), logits_cpu):
                        key = (condition, component, verb)
                        if key not in verb_token_logit_sums:
                            verb_token_logit_sums[key] = torch.zeros(len(target_df), dtype=torch.float64)
                            verb_token_logit_counts[key] = 0
                        verb_token_logit_sums[key] += row_logits
                        verb_token_logit_counts[key] += 1

        for batch_index, (_example_id, row) in enumerate(batch_df.iterrows()):
            accumulate_attention_patterns(
                attention_sums,
                attention_counts,
                component_specs,
                heads,
                "clean",
                clean_cache,
                batch_index,
                positions_for_condition(row, "clean"),
            )
            accumulate_attention_patterns(
                attention_sums,
                attention_counts,
                component_specs,
                heads,
                "corrupt",
                corrupt_cache,
                batch_index,
                positions_for_condition(row, "corrupt"),
            )
        del clean_cache, corrupt_cache

    base = {col: slot_row[col] for col in SLOT_COLUMNS}
    base["selected_budget"] = int(slot_row["selected_budget"])

    target_rows: list[dict[str, Any]] = []
    for component in components:
        spec = component_specs[component]
        for condition in ["clean", "corrupt"]:
            count = token_logit_counts[(condition, component)]
            mean_logits = token_logit_sums[(condition, component)] / max(count, 1)
            for target_index, target in enumerate(target_records):
                target_rows.append(
                    {
                        **base,
                        "component": component,
                        "component_type": spec["component_type"],
                        "layer": spec["layer"],
                        "head": spec["head"],
                        "condition": condition,
                        **target,
                        "mean_logit": float(mean_logits[target_index]),
                        "example_count": int(count),
                    }
                )

    verb_rows: list[dict[str, Any]] = []
    for condition, component, verb in sorted(verb_token_logit_sums.keys(), key=lambda item: (item[1], item[0], item[2])):
        spec = component_specs[component]
        count = verb_token_logit_counts[(condition, component, verb)]
        mean_logits = verb_token_logit_sums[(condition, component, verb)] / max(count, 1)
        for target_index, target in enumerate(target_records):
            verb_rows.append(
                {
                    **base,
                    "component": component,
                    "component_type": spec["component_type"],
                    "layer": spec["layer"],
                    "head": spec["head"],
                    "condition": condition,
                    "verb": verb,
                    **target,
                    "mean_logit": float(mean_logits[target_index]),
                    "example_count": int(count),
                }
            )

    metadata = pd.DataFrame(
        [
            {
                **base,
                "model_name": context.get("model_name"),
                "requested_model_name": context.get("requested_model_name"),
                "target_source": context.get("target_source"),
                "target_filter_policy": context.get("target_filter_policy"),
                "target_settings_source": context.get("target_settings_source"),
                "summary_path": context.get("summary_path"),
                "example_count": int(len(examples)),
                "component_count": int(len(components)),
                "attention_head_count": int(len(heads)),
                "animate_target_count": int((target_df["target_set"] == "animate").sum()),
                "inanimate_target_count": int((target_df["target_set"] == "inanimate").sum()),
            }
        ]
    )

    target_logit_lens = pd.DataFrame(target_rows)
    if str(slot_row["model_slug"]) == "gpt2":
        mlp_target_set_summary = mlp_prompt_target_set_logit_lens_summary(target_logit_lens)
    else:
        mlp_target_set_summary = pd.DataFrame()

    del context["model"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "target_logit_lens": target_logit_lens,
        "mlp_prompt_target_set_logit_lens": mlp_target_set_summary,
        "verb_target_logit_lens": pd.DataFrame(verb_rows),
        "attention_pattern_summary": attention_summary_from_accumulators(slot_row, heads, attention_sums, attention_counts),
        "target_metadata": metadata,
    }


def concat_result_table(results: list[dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    frames = [result[key] for result in results if key in result and isinstance(result[key], pd.DataFrame) and not result[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute necessary-circuit attention-pattern and target logit-lens diagnostics.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-limit", type=int, default=80, help="Maximum single-token targets per target set. Use -1 for all.")
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
    output_dir = args.output_dir or (results_root / f"necessary_circuit_diagnostics_{stamp}")
    seeds = None if args.all_seeds else args.seeds
    target_limit = None if args.target_limit is not None and args.target_limit < 0 else args.target_limit

    selected_summary, semantic_edges, component_cards = load_report(report_dir, args.model_slug, seeds)
    del semantic_edges
    prompt_pairs = None if args.no_prompt_intersection else build_prompt_intersection(project_root, selected_summary, args.batch_size)

    results: list[dict[str, pd.DataFrame]] = []
    completed_slots: list[str] = []
    slot_batch_sizes: dict[str, int] = {}
    base_metadata = {
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "n_examples": int(args.n_examples),
        "batch_size_override": int(args.batch_size) if args.batch_size is not None else None,
        "target_limit_per_set": target_limit,
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
            f"with n_examples={args.n_examples}, batch_size={slot_batch_size}, target_limit={target_limit}"
        )
        results.append(
            analyze_slot(
                project_root=project_root,
                component_cards=component_cards,
                slot_row=slot_row,
                n_examples=args.n_examples,
                batch_size=slot_batch_size,
                target_limit=target_limit,
                prompt_pairs=prompt_pairs,
            )
        )
        completed_slots.append(slot_key)
        tables = {
            "target_logit_lens": concat_result_table(results, "target_logit_lens"),
            "mlp_prompt_target_set_logit_lens": concat_result_table(results, "mlp_prompt_target_set_logit_lens"),
            "verb_target_logit_lens": concat_result_table(results, "verb_target_logit_lens"),
            "attention_pattern_summary": concat_result_table(results, "attention_pattern_summary"),
            "target_metadata": concat_result_table(results, "target_metadata"),
        }
        metadata = {
            **base_metadata,
            "slot_batch_sizes": slot_batch_sizes,
            "completed_slots": completed_slots,
            "completed_slot_count": int(len(completed_slots)),
            "table_rows": {name: int(len(frame)) for name, frame in tables.items()},
        }
        write_outputs(output_dir, tables, metadata)
        print(f"Checkpointed diagnostics to {output_dir}")

    print(f"Wrote necessary-circuit diagnostics to {output_dir}")


if __name__ == "__main__":
    main()

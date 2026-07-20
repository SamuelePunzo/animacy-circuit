from __future__ import annotations

import argparse
from collections import Counter
import gc
from pathlib import Path

CONTROL_NAME = "verb_noise_calibration"
DEFAULT_SIGMA_MULTIPLIERS = (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run per-model Gaussian verb-noise sigma calibration and save sweep artifacts.",
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset-filter-model", default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-batch-size", type=int, default=50)
    parser.add_argument(
        "--target-filter-policy",
        choices=("none", "recovery_margin", "model_success"),
        default="model_success",
    )
    parser.add_argument("--noise-site", default="hook_resid_pre")
    parser.add_argument(
        "--sigma-multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_SIGMA_MULTIPLIERS),
    )
    parser.add_argument(
        "--output-day",
        default=None,
        help="Results day tag. Defaults to today's tag when omitted.",
    )
    return parser


def inspect_verb_noise_token_validation(df, tokenizer) -> dict[str, object]:
    from circuit_finder_core import (
        normalize_concept_pair_metadata,
        pair_token_alignment_details,
    )

    normalized = normalize_concept_pair_metadata(df)
    metadata_available = {"patient", "clean_verb", "corrupt_verb"}.issubset(normalized.columns)
    counts: Counter[str] = Counter()
    preview: list[dict[str, object]] = []

    for row_idx, row in normalized.reset_index(drop=True).iterrows():
        counts["input_rows"] += 1
        details = pair_token_alignment_details(
            row,
            tokenizer,
            metadata_available=metadata_available,
        )
        if not details["pair_ok"]:
            counts["pair_invalid_rows"] += 1
            clean_error = str(details["clean_verb_error"] or "none")
            corrupt_error = str(details["corrupt_verb_error"] or "none")
            counts[f"pair_invalid::{clean_error}|{corrupt_error}"] += 1
            if len(preview) < 5:
                preview.append(
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

        counts["pair_ok_rows"] += 1
        clean_span = details["clean_verb_span"]
        corrupt_span = details["corrupt_verb_span"]
        assert clean_span is not None
        assert corrupt_span is not None
        clean_width = int(clean_span[1] - clean_span[0])
        corrupt_width = int(corrupt_span[1] - corrupt_span[0])

        if clean_width == 1 and corrupt_width == 1:
            counts["single_token_verb_rows"] += 1
            continue

        counts["single_token_failure_rows"] += 1
        counts[f"single_token_failure::{clean_width}|{corrupt_width}"] += 1
        if len(preview) < 5:
            preview.append(
                {
                    "row": int(row_idx),
                    "clean_prefix": details["clean_prefix"],
                    "corrupt_prefix": details["corrupt_prefix"],
                    "clean_verb": details["clean_verb"],
                    "corrupt_verb": details["corrupt_verb"],
                    "clean_verb_span": clean_span,
                    "corrupt_verb_span": corrupt_span,
                    "clean_width": clean_width,
                    "corrupt_width": corrupt_width,
                    "clean_verb_error": "verb_not_single_token",
                    "corrupt_verb_error": "verb_not_single_token",
                }
            )

    return {
        "input_rows": int(counts["input_rows"]),
        "pair_ok_rows": int(counts["pair_ok_rows"]),
        "pair_invalid_rows": int(counts["pair_invalid_rows"]),
        "single_token_verb_rows": int(counts["single_token_verb_rows"]),
        "single_token_failure_rows": int(counts["single_token_failure_rows"]),
        "failure_reason_counts": dict(sorted(counts.items())),
        "failure_preview": preview,
    }


def run_calibration(
    *,
    model_name: str,
    dataset_filter_model_name: str,
    seed: int,
    filter_batch_size: int,
    target_filter_policy: str,
    noise_site: str,
    sigma_multipliers: list[float],
    output_day: str | None,
) -> dict[str, object]:
    import torch

    from circuit_finder_core import (
        add_sequence_lengths,
        canonical_model_name,
        DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS,
        concept_hook_name,
        date_tag,
        find_metric_filtered_model_dataset_path,
        load_metric_filtered_model_success_dataset,
        load_model_context,
        resolve_animacy_circuit_root,
        save_csv,
        save_json,
    )
    from control_runners import (
        add_control_pair_seeds,
        control_output_dir,
        export_selected_sigma,
        prepare_verb_noise_control_rows,
        select_verb_noise_sigma,
        sweep_verb_noise_sigmas,
    )

    project_root = resolve_animacy_circuit_root(Path.cwd())
    resolved_model_name = canonical_model_name(model_name)
    hook_name = concept_hook_name(0, noise_site)
    resolved_day = output_day or date_tag()
    output_dir = control_output_dir(
        project_root=project_root,
        model_name=resolved_model_name,
        day=resolved_day,
        control_name=CONTROL_NAME,
    )
    precheck_path = output_dir / "calibration_precheck.json"
    if target_filter_policy != "model_success":
        raise ValueError(
            "Verb-noise calibration currently supports only "
            "target_filter_policy='model_success' because it loads the saved per-model filtered dataset."
        )

    metric_filtered_path = find_metric_filtered_model_dataset_path(
        project_root,
        resolved_model_name,
    )
    if metric_filtered_path is None:
        raise FileNotFoundError(
            f"No saved metric-filtered dataset found for {resolved_model_name}."
        )
    print(
        f"Loading saved metric-filtered dataset for {resolved_model_name} "
        f"from {metric_filtered_path}"
    )

    filtered_df = load_metric_filtered_model_success_dataset(
        project_root=project_root,
        model_name=resolved_model_name,
        path=metric_filtered_path,
        common_filter_model_names=DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS,
    )
    context = load_model_context(
        project_root,
        resolved_model_name,
        target_filter_model_names=DEFAULT_COMMON_TOKENIZATION_FILTER_MODELS,
    )
    print(f"Target-filtered rows: {len(filtered_df)}")

    token_validation = inspect_verb_noise_token_validation(
        filtered_df,
        context["tokenizer"],
    )
    print(
        "Verb-noise token validation: "
        f"input_rows={token_validation['input_rows']}, "
        f"pair_ok_rows={token_validation['pair_ok_rows']}, "
        f"single_token_verb_rows={token_validation['single_token_verb_rows']}, "
        f"pair_invalid_rows={token_validation['pair_invalid_rows']}, "
        f"single_token_failure_rows={token_validation['single_token_failure_rows']}"
    )
    if int(token_validation["single_token_verb_rows"]) == 0:
        failure_payload = {
            "model_name": resolved_model_name,
            "dataset_filter_model_name": dataset_filter_model_name,
            "seed": int(seed),
            "filter_batch_size": int(filter_batch_size),
            "target_filter_policy": target_filter_policy,
            "noise_site": noise_site,
            "hook_name": hook_name,
            "output_day": resolved_day,
            "metric_filtered_path": str(metric_filtered_path),
            "stage": "token_validation",
            "target_filtered_count": int(len(filtered_df)),
            "token_validation": token_validation,
        }
        save_json(precheck_path, failure_payload)
        raise ValueError(
            "No rows survived verb-noise single-token validation. "
            f"Diagnostics saved to {precheck_path}. "
            f"Counts: {token_validation}"
        )

    validated_rows = prepare_verb_noise_control_rows(
        filtered_df,
        context["tokenizer"],
    )
    with_lengths = add_sequence_lengths(validated_rows, context["model"])
    print(
        "Verb-noise sequence-length validation: "
        f"single_token_verb_rows={len(validated_rows)}, "
        f"sequence_length_ok_rows={len(with_lengths)}"
    )
    if len(with_lengths) != len(validated_rows):
        failure_payload = {
            "model_name": resolved_model_name,
            "dataset_filter_model_name": dataset_filter_model_name,
            "seed": int(seed),
            "filter_batch_size": int(filter_batch_size),
            "target_filter_policy": target_filter_policy,
            "noise_site": noise_site,
            "hook_name": hook_name,
            "output_day": resolved_day,
            "metric_filtered_path": str(metric_filtered_path),
            "stage": "sequence_length_validation",
            "target_filtered_count": int(len(filtered_df)),
            "token_validation": token_validation,
            "single_token_verb_rows": int(len(validated_rows)),
            "sequence_length_ok_rows": int(len(with_lengths)),
        }
        save_json(precheck_path, failure_payload)
        raise ValueError(
            "Verb-noise control lost rows during sequence-length validation. "
            f"Diagnostics saved to {precheck_path}. "
            f"single_token_verb_rows={len(validated_rows)}, "
            f"sequence_length_ok_rows={len(with_lengths)}"
        )

    control_df = add_control_pair_seeds(with_lengths, seed).reset_index(drop=True).copy()
    sweep = sweep_verb_noise_sigmas(
        df=control_df,
        model=context["model"],
        animate_ids_tensor=context["animate_ids_tensor"],
        inanimate_ids_tensor=context["inanimate_ids_tensor"],
        batch_size=filter_batch_size,
        sigma_multipliers=sigma_multipliers,
        hook_name=hook_name,
    )

    sweep_df = (
        sweep["sweep_df"]
        .copy()
        .sort_values("sigma_multiplier")
        .reset_index(drop=True)
    )
    selected = select_verb_noise_sigma(sweep_df)
    original_summary = dict(sweep["original_summary"])
    original_absolute_margin_mean = abs(float(original_summary["margin_mean"]))
    sweep_path = output_dir / "sigma_sweep.csv"
    summary_path = output_dir / "calibration_summary.json"
    selected_path = output_dir / "selected_sigma.json"

    save_csv(sweep_df, sweep_path, index=False)

    summary = {
        "model_name": resolved_model_name,
        "dataset_filter_model_name": dataset_filter_model_name,
        "seed": int(seed),
        "filter_batch_size": int(filter_batch_size),
        "target_filter_policy": target_filter_policy,
        "noise_site": noise_site,
        "hook_name": hook_name,
        "output_day": resolved_day,
        "metric_filtered_path": str(metric_filtered_path),
        "sigma_multipliers": [float(value) for value in sigma_multipliers],
        "target_filtered_count": int(len(filtered_df)),
        "validated_control_rows": int(len(control_df)),
        "token_validation": token_validation,
        "activation_scale": float(sweep["activation_scale"]),
        "original_summary": original_summary,
        "original_absolute_margin_mean": float(original_absolute_margin_mean),
        "selected": selected,
        "paths": {
            "sigma_sweep_csv": str(sweep_path),
            "selected_sigma_json": str(selected_path),
        },
    }
    save_json(summary_path, summary)

    export_selected_sigma(
        selected_path,
        {
            "model_name": resolved_model_name,
            "dataset_filter_model_name": dataset_filter_model_name,
            "seed": int(seed),
            "filter_batch_size": int(filter_batch_size),
            "target_filter_policy": target_filter_policy,
            "noise_site": noise_site,
            "hook_name": hook_name,
            "output_day": resolved_day,
            "metric_filtered_path": str(metric_filtered_path),
            "target_filtered_count": int(len(filtered_df)),
            "validated_control_rows": int(len(control_df)),
            "activation_scale": float(sweep["activation_scale"]),
            "original_margin_mean": float(original_summary["margin_mean"]),
            "original_absolute_margin_mean": float(original_absolute_margin_mean),
            **selected,
            "calibration_summary_path": str(summary_path),
            "sigma_sweep_csv": str(sweep_path),
        },
    )

    del context
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "selected_path": str(selected_path),
        "sweep_path": str(sweep_path),
        "selected_sigma": float(selected["sigma"]),
        "selected_sigma_multiplier": float(selected["sigma_multiplier"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    artifact = run_calibration(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        seed=args.seed,
        filter_batch_size=args.filter_batch_size,
        target_filter_policy=args.target_filter_policy,
        noise_site=args.noise_site,
        sigma_multipliers=[float(value) for value in args.sigma_multipliers],
        output_day=args.output_day,
    )
    print(f"Calibration saved to {artifact['output_dir']}")
    print(
        "Selected sigma "
        f"{artifact['selected_sigma']:.6g} "
        f"(multiplier={artifact['selected_sigma_multiplier']:.6g})"
    )
    print(f"Summary: {artifact['summary_path']}")
    print(f"Selected sigma JSON: {artifact['selected_path']}")
    print(f"Sweep CSV: {artifact['sweep_path']}")


if __name__ == "__main__":
    main()

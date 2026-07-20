from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXPERIMENTS = {
    "prepare-tokenization-filters",
    "metric-investigation-score",
    "prepare-dataset",
    "tokenization-check",
    "diagnose-model",
    "component-discovery",
    "concept-extraction",
    "eap-full",
    "eap-selected",
    "eap-shadow-rediscovery",
    "eap-localization",
    "eap-conditional-ablation",
    "dual-set-eap",
    "control-verb-noise",
    "control-by-to-near",
    "control-blimp-passive-prefix",
    "compare",
}


def add_target_source_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-source",
        default="wordnet",
        help=(
            "Target set source: 'wordnet', 'abstract_agency', or a path to a "
            "target JSON with targets.animate and targets.inanimate."
        ),
    )


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="gpt2",
        help=(
            "Model to inspect. Short aliases such as 'Llama 3.2 3B', "
            "'Qwen 3 4B', and 'Gemma 3 4B base' are resolved to HF IDs."
        ),
    )
    parser.add_argument(
        "--dataset-filter-model",
        default="gpt2",
        help="Model used to define the shared success-filtered prompt slice.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discovery-sample-size", type=int, default=500)
    parser.add_argument(
        "--discovery-margin-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum clean-corrupt metric margin for examples eligible for the "
            "discovery sample. The retained dataset itself is only sign-filtered."
        ),
    )
    parser.add_argument("--filter-batch-size", type=int, default=50)
    parser.add_argument("--output-day", default=None)
    parser.add_argument("--dataset-filter-path", default=None)
    parser.add_argument("--refresh-dataset-filter", action="store_true")
    parser.add_argument(
        "--cache-dataset-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache the source model success pool to disk. By default it is "
            "loaded from or saved to results/model_success."
        ),
    )
    parser.add_argument("--max-filter-examples", type=int, default=None)
    parser.add_argument(
        "--target-filter-policy",
        choices=("none", "recovery_margin", "model_success"),
        default="model_success",
        help=(
            "How to filter the shared success slice after scoring it with the inspected model. "
            "model_success keeps examples with clean_metric > 0 and corrupt_metric < 0."
        ),
    )
    parser.add_argument("--start-path", default=None)
    add_target_source_arg(parser)


def add_eap_args(parser: argparse.ArgumentParser) -> None:
    add_common_model_args(parser)
    parser.add_argument("--attribution-batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Collapsed-edge budget sweep. When omitted, uses a fixed low-budget prefix plus a geometric tail.",
    )
    parser.add_argument(
        "--budget-max-fraction",
        type=float,
        default=0.15,
        help="Alpha in k_max = max(k_floor, ceil(alpha * ranked_edge_count)) for generated EAP budgets.",
    )
    parser.add_argument(
        "--budget-floor",
        type=int,
        default=2000,
        help="k_floor in k_max = max(k_floor, ceil(alpha * ranked_edge_count)) for generated EAP budgets.",
    )
    parser.add_argument(
        "--budget-tail-points",
        type=int,
        default=20,
        help="Number of geometric tail points between 300 and the generated k_max.",
    )
    parser.add_argument(
        "--budget-early-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop the EAP budget sweep once faithfulness reaches the configured threshold.",
    )
    parser.add_argument(
        "--budget-early-stop-threshold",
        type=float,
        default=0.85,
        help="Faithfulness threshold that terminates the sweep when reached.",
    )
    parser.add_argument(
        "--budget-early-stop-patience",
        type=int,
        default=5,
        help="Legacy no-op retained for CLI compatibility with older runs.",
    )
    parser.add_argument(
        "--budget-early-stop-min-delta",
        type=float,
        default=0.01,
        help="Legacy no-op retained for CLI compatibility with older runs.",
    )
    parser.add_argument(
        "--budget-early-stop-start-budget",
        type=int,
        default=300,
        help="Legacy no-op retained for CLI compatibility with older runs.",
    )


def add_dual_set_args(parser: argparse.ArgumentParser) -> None:
    add_eap_args(parser)
    for action in parser._actions:
        if action.dest == "dataset_filter_model":
            action.help = argparse.SUPPRESS
            break
    parser.add_argument(
        "--shared-filter-models",
        nargs="+",
        required=True,
        help=(
            "Models whose retained sets define the shared intersection. "
            "The target model is included automatically."
        ),
    )
    parser.add_argument(
        "--dataset-sets",
        nargs="+",
        choices=("model_specific_correct", "shared_correct"),
        default=("model_specific_correct", "shared_correct"),
        help=(
            "Dataset sets to run. Defaults to both; use model_specific_correct "
            "to skip the shared/intersection set."
        ),
    )
    parser.add_argument(
        "--run-diagnose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run dataset-set diagnostics before EAP.",
    )
    parser.add_argument(
        "--run-eap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run full-model EAP-IG on each dataset set.",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single entry point for animacy circuit experiments."
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    tokenization_prepare = subparsers.add_parser(
        "prepare-tokenization-filters",
        help="Filter minimal pairs and target sets by model tokenizer single-token components.",
    )
    tokenization_prepare.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Models whose tokenizers define model-specific filters and the shared "
            "intersection. Defaults to the project ensemble."
        ),
    )
    tokenization_prepare.add_argument("--refresh", action="store_true")
    tokenization_prepare.add_argument("--start-path", default=None)
    add_target_source_arg(tokenization_prepare)

    metric_score = subparsers.add_parser(
        "metric-investigation-score",
        help="GPU scoring for the metric investigation notebook section 3.",
    )
    metric_score.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Models to score on the common ensemble-tokenizer-filtered dataset. "
            "Defaults to the project ensemble."
        ),
    )
    metric_score.add_argument("--batch-size", type=int, default=32)
    metric_score.add_argument("--top-k", type=int, default=50)
    metric_score.add_argument(
        "--margin-threshold",
        type=float,
        default=0.5,
        help=(
            "Margin threshold used only to save mp_discovery_candidates_<metric>.csv. "
            "The mp_filtered_<metric>.csv datasets are sign-filtered only."
        ),
    )
    metric_score.add_argument("--refresh-tokenization-filters", action="store_true")
    metric_score.add_argument("--output-day", default=None)
    metric_score.add_argument("--start-path", default=None)
    add_target_source_arg(metric_score)

    prepare = subparsers.add_parser(
        "prepare-dataset",
        help="Build or refresh the shared model-success dataset cache.",
    )
    prepare.add_argument(
        "--dataset-filter-model",
        default="gpt2",
        help="Model used to define the shared success-filtered prompt slice.",
    )
    prepare.add_argument("--filter-batch-size", type=int, default=50)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dataset-filter-path", default=None)
    prepare.add_argument("--refresh-dataset-filter", action="store_true")
    prepare.add_argument("--max-filter-examples", type=int, default=None)
    prepare.add_argument("--start-path", default=None)
    add_target_source_arg(prepare)

    tokenization = subparsers.add_parser(
        "tokenization-check",
        help="Count tokenizer alignment and target-set tokenization failures for a model.",
    )
    tokenization.add_argument(
        "--model",
        default="gpt2",
        help=(
            "Model tokenizer to inspect. Short aliases such as 'Llama 3.2 3B', "
            "'Qwen 3 4B', and 'Gemma 3 4B base' are resolved to HF IDs."
        ),
    )
    tokenization.add_argument("--start-path", default=None)
    add_target_source_arg(tokenization)

    diagnostic = subparsers.add_parser(
        "diagnose-model",
        help="Run tokenization and task-accuracy diagnostics without circuit experiments.",
    )
    diagnostic.add_argument(
        "--model",
        default="gpt2",
        help=(
            "Model to inspect. Short aliases such as 'Llama 3.2 3B', "
            "'Qwen 3 4B', and 'Gemma 3 4B base' are resolved to HF IDs."
        ),
    )
    diagnostic.add_argument(
        "--dataset-filter-model",
        default="gpt2",
        help="Model used to define the shared success-filtered prompt slice.",
    )
    diagnostic.add_argument("--filter-batch-size", type=int, default=50)
    diagnostic.add_argument("--seed", type=int, default=42)
    diagnostic.add_argument("--dataset-filter-path", default=None)
    diagnostic.add_argument("--refresh-dataset-filter", action="store_true")
    diagnostic.add_argument(
        "--cache-dataset-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache the source model success pool to disk. Default is to load "
            "from or save to results/model_success."
        ),
    )
    diagnostic.add_argument("--max-filter-examples", type=int, default=None)
    diagnostic.add_argument(
        "--target-filter-policy",
        choices=("none", "recovery_margin", "model_success"),
        default="model_success",
        help="How to filter the shared success slice after scoring it with the inspected model.",
    )
    diagnostic.add_argument("--output-day", default=None)
    diagnostic.add_argument("--output-dir", default=None)
    diagnostic.add_argument(
        "--save",
        action="store_true",
        help="Save the diagnostic summary JSON. Default is to print only.",
    )
    diagnostic.add_argument(
        "--save-details",
        action="store_true",
        help="Also save the filtered diagnostic CSV alongside the summary JSON.",
    )
    diagnostic.add_argument(
        "--save-debug-details",
        action="store_true",
        help="Also save the raw and source-scored diagnostic CSVs for debugging.",
    )
    diagnostic.add_argument("--start-path", default=None)
    add_target_source_arg(diagnostic)

    component = subparsers.add_parser(
        "component-discovery",
        help="Run patching-based component discovery on the shared prompt slice.",
    )
    add_common_model_args(component)
    component.add_argument("--patch-batch-size", type=int, default=16)
    component.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=None,
        help="Residual/module percentile thresholds.",
    )

    concept = subparsers.add_parser(
        "concept-extraction",
        help="Extract and validate an animacy concept vector from residual-stream verb activations.",
    )
    add_common_model_args(concept)
    concept.add_argument(
        "--max-concept-examples",
        type=int,
        default=None,
        help="Optionally subsample the already filtered concept dataset before the 60/20/20 split.",
    )
    concept.add_argument("--extraction-batch-size", type=int, default=16)
    concept.add_argument("--steering-batch-size", type=int, default=16)
    concept.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Steering strengths to sweep on validation. Defaults to "
            "-10 -7.5 -5 -3 -2 -1 0 1 2 3 5 7.5 10."
        ),
    )
    concept.add_argument(
        "--hook-points",
        nargs="+",
        default=None,
        choices=("hook_resid_pre", "hook_resid_mid", "hook_resid_post"),
        help="Residual stream hook points to evaluate across all layers.",
    )
    concept.add_argument(
        "--normalize-concept-vector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each concept vector to unit norm before steering. Default: true.",
    )
    concept.add_argument(
        "--selection-effect-fraction",
        type=float,
        default=0.90,
        help="Select the smallest |alpha| whose validation effect is at least this fraction of the best.",
    )
    concept.add_argument(
        "--random-control-repeats",
        type=int,
        default=10,
        help="Number of random direction controls to evaluate on the test split at the selected hook/alpha.",
    )

    eap_full = subparsers.add_parser(
        "eap-full",
        help="Run full-model EAP-IG on the shared prompt slice.",
    )
    add_eap_args(eap_full)

    eap_selected = subparsers.add_parser(
        "eap-selected",
        help="Run selected-component EAP-IG using saved component-discovery outputs.",
    )
    add_eap_args(eap_selected)
    eap_selected.add_argument("--circuit-finder-day", default=None)
    eap_selected.add_argument("--importance-quantile", type=float, default=0.10)
    eap_selected.add_argument(
        "--component-threshold",
        type=int,
        default=None,
        help=(
            "Component-discovery percentile threshold to use for selected EAP. "
            "Defaults to the highest available threshold with retained nodes."
        ),
    )

    eap_shadow = subparsers.add_parser(
        "eap-shadow-rediscovery",
        help="Remove saved full-model EAP edges and rerun EAP-IG on the remaining graph.",
    )
    add_eap_args(eap_shadow)
    eap_shadow.add_argument(
        "--dataset-set",
        default="model_specific_correct",
        choices=("model_specific_correct", "shared_correct"),
        help="Dataset set whose saved full-model EAP run should define the source circuit.",
    )
    eap_shadow.add_argument(
        "--shared-filter-models",
        nargs="+",
        default=["gpt2"],
        help="Models used to define the shared-correct set when --dataset-set shared_correct is used.",
    )
    eap_shadow.add_argument(
        "--main-experiment-path",
        default=None,
        help=(
            "Optional path to a saved full-model EAP artifact: a full_model directory, "
            "full_model_summary_*.json, full_model_edges_*.csv, or full_model_budget_sweep_*.csv. "
            "When omitted, the latest matching saved full-model run is used."
        ),
    )
    eap_shadow.add_argument(
        "--source-faithfulness-threshold",
        type=float,
        default=0.85,
        help="Threshold used to choose the source circuit budget for the whole-circuit removal variant.",
    )
    eap_shadow.add_argument(
        "--top-edge-count",
        type=int,
        default=100,
        help="Number of highest-ranked collapsed source edges to remove in the top-k variant.",
    )
    eap_shadow.add_argument(
        "--variants",
        nargs="+",
        choices=("top_k", "first_85pct"),
        default=None,
        help=(
            "Which source-circuit removal variants to run. Defaults to both; use "
            "`--variants top_k` to run only the top-k removal."
        ),
    )
    eap_shadow.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing shadow rediscovery edge/node rankings when present.",
    )

    eap_localization = subparsers.add_parser(
        "eap-localization",
        help="Run EAP-IG localization/distribution diagnostics across sample sizes and seeds.",
    )
    add_eap_args(eap_localization)
    eap_localization.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000, 2000],
        help="Discovery sample sizes to run.",
    )
    eap_localization.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 42],
        help="Random seeds for independently sampled discovery batches.",
    )
    eap_localization.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        help="Collapsed-edge budgets for keep-only, ablate, and random-baseline evaluation.",
    )
    eap_localization.add_argument("--random-repeats", type=int, default=10)
    eap_localization.add_argument(
        "--max-validation-examples",
        type=int,
        default=None,
        help=(
            "Optional cap on held-out examples evaluated for each graph. "
            "Use this for pilot runs; omit for full held-out evaluation."
        ),
    )
    eap_localization.add_argument("--skip-random-baselines", action="store_true")
    eap_localization.add_argument("--skip-existing", action="store_true")
    eap_localization.add_argument(
        "--import-edge-rankings-path",
        default=None,
        help=(
            "Optional path to a precomputed collapsed-edge ranking CSV, such as "
            "full_model_edges_<day>.csv from a prior EAP-IG run. When provided, "
            "localization reuses that ranking for one specific (sample_size, seed) slot "
            "instead of recomputing attribution."
        ),
    )
    eap_localization.add_argument(
        "--import-summary-path",
        default=None,
        help=(
            "Optional summary JSON for the imported ranking. When present, localization "
            "verifies model/filter/sample provenance before reusing the ranking."
        ),
    )
    eap_localization.add_argument(
        "--import-sample-size",
        type=int,
        default=None,
        help="Discovery sample size slot that should reuse the imported ranking.",
    )
    eap_localization.add_argument(
        "--import-seed",
        type=int,
        default=None,
        help="Seed slot that should reuse the imported ranking.",
    )

    conditional_ablation = subparsers.add_parser(
        "eap-conditional-ablation",
        help="Compare top-k ablation against disjoint high-importance edge-set ablations.",
    )
    add_common_model_args(conditional_ablation)
    conditional_ablation.add_argument(
        "--sample-size",
        dest="discovery_sample_size",
        type=int,
        default=500,
        help="Alias for --discovery-sample-size used to choose the localization slot.",
    )
    conditional_ablation.add_argument("--evaluation-batch-size", type=int, default=1)
    conditional_ablation.add_argument(
        "--max-validation-examples",
        type=int,
        default=None,
        help="Optional cap on held-out examples evaluated for each ablation set.",
    )
    conditional_ablation.add_argument(
        "--dataset-mode",
        choices=("semantic_filtered", "named_entity_truncated"),
        default="semantic_filtered",
    )
    conditional_ablation.add_argument("--dataset-set", default="model_specific_correct")
    conditional_ablation.add_argument("--named-entity-discovery-dir", default=None)
    conditional_ablation.add_argument("--target-token-mode", default="first_token")
    conditional_ablation.add_argument(
        "--localization-source-path",
        default=None,
        help=(
            "Path to a localization slot summary, localization manifest, or localization "
            "output directory. When provided, the analysis reuses that ranking and slot."
        ),
    )
    conditional_ablation.add_argument(
        "--edge-rankings-path",
        default=None,
        help="Fallback collapsed-edge ranking CSV when no localization source is provided.",
    )
    conditional_ablation.add_argument("--protected-budget", type=int, default=20)
    conditional_ablation.add_argument("--ablated-budget", type=int, default=20)
    conditional_ablation.add_argument("--candidate-start-rank", type=int, default=21)
    conditional_ablation.add_argument("--candidate-end-rank", type=int, default=200)
    conditional_ablation.add_argument(
        "--band-size",
        type=int,
        default=None,
        help="Rank-band width. Defaults to the ablated budget.",
    )
    conditional_ablation.add_argument("--sample-count", type=int, default=100)
    conditional_ablation.add_argument(
        "--sampling-strategy",
        choices=("score_weighted", "uniform"),
        default="score_weighted",
    )
    conditional_ablation.add_argument("--random-seed", type=int, default=0)

    dual_set = subparsers.add_parser(
        "dual-set-eap",
        help="Run diagnose-model and/or full-model EAP-IG on both model-specific and shared-correct sets.",
    )
    add_dual_set_args(dual_set)

    control_noise = subparsers.add_parser(
        "control-verb-noise",
        help="Cross-evaluate a saved full-model EAP ranking on the Gaussian verb-noise control.",
    )
    add_common_model_args(control_noise)
    control_noise.add_argument("--evaluation-batch-size", type=int, default=1)
    control_noise.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Scalar Gaussian noise level chosen externally, typically from the calibration notebook.",
    )
    control_noise.add_argument(
        "--noise-site",
        default="hook_resid_pre",
        choices=("hook_resid_pre",),
        help="Residual stream hook point for the verb-noise intervention.",
    )
    control_noise.add_argument(
        "--main-experiment-path",
        default=None,
        help=(
            "Path to a saved full-model main artifact: a full_model_summary_*.json, "
            "full_model_edges_*.csv, full_model_budget_sweep_*.csv, or the containing full_model directory."
        ),
    )
    control_noise.add_argument(
        "--max-budgets",
        type=int,
        default=None,
        help="Evaluate only the first N matched budgets. Leave unset for the full matched-budget sweep.",
    )
    control_noise.add_argument(
        "--run-second-stage-discovery-on-ambiguous",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If the control looks ambiguous, run noisy full-model EAP-IG on the original discovery rows.",
    )

    control_near = subparsers.add_parser(
        "control-by-to-near",
        help="Cross-evaluate a saved full-model EAP ranking on the by->near control.",
    )
    add_common_model_args(control_near)
    control_near.add_argument("--evaluation-batch-size", type=int, default=1)
    control_near.add_argument(
        "--replacement-from",
        default=" by the",
        help="Terminal substring to replace in clean/corrupt prefixes.",
    )
    control_near.add_argument(
        "--replacement-to",
        default=" near the",
        help="Replacement terminal substring for the control prefixes.",
    )
    control_near.add_argument(
        "--main-experiment-path",
        default=None,
        help=(
            "Path to a saved full-model main artifact: a full_model_summary_*.json, "
            "full_model_edges_*.csv, full_model_budget_sweep_*.csv, or the containing full_model directory."
        ),
    )
    control_near.add_argument(
        "--max-budgets",
        type=int,
        default=None,
        help="Evaluate only the first N matched budgets. Leave unset for the full matched-budget sweep.",
    )
    control_near.add_argument(
        "--run-second-stage-discovery-on-ambiguous",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If the control looks ambiguous, run full-model EAP-IG on the rewritten discovery rows.",
    )

    control_blimp_prefix = subparsers.add_parser(
        "control-blimp-passive-prefix",
        help="Compare the full model and a retained source circuit on BLiMP animate_subject_passive prefixes using the repo animate/inanimate target sets.",
    )
    add_common_model_args(control_blimp_prefix)
    control_blimp_prefix.add_argument("--evaluation-batch-size", type=int, default=32)
    control_blimp_prefix.add_argument(
        "--main-experiment-path",
        default=None,
        help=(
            "Path to a saved full-model main artifact: a full_model_summary_*.json, "
            "full_model_edges_*.csv, full_model_budget_sweep_*.csv, or the containing full_model directory."
        ),
    )
    control_blimp_prefix.add_argument(
        "--source-faithfulness-threshold",
        type=float,
        default=0.85,
        help="Use the first source full-model budget whose saved faithfulness reaches this value.",
    )
    control_blimp_prefix.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Explicit collapsed-edge budget override for the retained circuit.",
    )

    subparsers.add_parser(
        "compare",
        help="Compare saved full-model and selected-component budget sweeps.",
    )
    return parser


def path_or_none(value: str | None) -> Path | None:
    return Path(value) if value is not None else None


def run_prepare_dataset(args: argparse.Namespace) -> None:
    from circuit_finder_core import load_or_create_model_success_dataset, model_success_dataset_path
    from utils import canonical_model_name, resolve_animacy_circuit_root

    project_root = resolve_animacy_circuit_root(path_or_none(args.start_path))
    resolved_model_name = canonical_model_name(args.dataset_filter_model)
    cache_path = (
        Path(args.dataset_filter_path)
        if args.dataset_filter_path is not None
        else model_success_dataset_path(
            project_root,
            resolved_model_name,
            target_source=args.target_source,
        )
    )
    df = load_or_create_model_success_dataset(
        project_root=project_root,
        model_name=resolved_model_name,
        batch_size=args.filter_batch_size,
        cache_path=cache_path,
        refresh=args.refresh_dataset_filter,
        cache=True,
        max_examples=args.max_filter_examples,
        seed=args.seed,
        target_source=args.target_source,
    )
    print(
        f"Prepared {len(df)} {resolved_model_name} success examples "
        f"at {cache_path}"
    )


def run_prepare_tokenization_filters(args: argparse.Namespace) -> None:
    from circuit_finder_core import (
        DEFAULT_TOKENIZATION_FILTER_MODELS,
        prepare_tokenization_filter_artifacts,
    )
    from utils import resolve_animacy_circuit_root

    project_root = resolve_animacy_circuit_root(path_or_none(args.start_path))
    model_names = tuple(args.models) if args.models is not None else DEFAULT_TOKENIZATION_FILTER_MODELS
    artifact = prepare_tokenization_filter_artifacts(
        project_root=project_root,
        model_names=model_names,
        refresh=args.refresh,
        target_source=args.target_source,
    )
    summary = artifact["summary"]
    print(f"Saved tokenization filter summary to {artifact['paths']['summary']}")
    for model_name, model_summary in summary["models"].items():
        print(
            f"{model_name}: {model_summary['pair_count']} pairs, "
            f"{model_summary['accepted_pair_count']} accepted JSONL pairs, "
            f"{model_summary['target_counts']['animate']} animate targets, "
            f"{model_summary['target_counts']['inanimate']} inanimate targets"
        )
    intersection = summary["intersection"]
    print(
        f"Intersection: {intersection['pair_count']} pairs, "
        f"{intersection['accepted_pair_count']} accepted JSONL pairs, "
        f"{intersection['target_counts']['animate']} animate targets, "
        f"{intersection['target_counts']['inanimate']} inanimate targets"
    )


def run_metric_investigation_score(args: argparse.Namespace) -> None:
    from circuit_finder_core import (
        DEFAULT_TOKENIZATION_FILTER_MODELS,
        run_metric_investigation_scoring,
    )
    from utils import resolve_animacy_circuit_root

    project_root = resolve_animacy_circuit_root(path_or_none(args.start_path))
    model_names = tuple(args.models) if args.models is not None else DEFAULT_TOKENIZATION_FILTER_MODELS
    artifact = run_metric_investigation_scoring(
        project_root=project_root,
        model_names=model_names,
        batch_size=args.batch_size,
        top_k=args.top_k,
        margin_threshold=args.margin_threshold,
        refresh_tokenization_filters=args.refresh_tokenization_filters,
        output_day=args.output_day,
        target_source=args.target_source,
    )
    print(f"Saved metric-investigation scoring summary to {artifact['paths']['summary']}")
    for model_name, model_summary in artifact["model_results"].items():
        if model_summary["status"] == "scored":
            print(
                f"{model_name}: scored {model_summary['scored_count']} rows; "
                f"success {model_summary['success_count']}; "
                f"cache {model_summary['paths']['model_success_cache']}"
            )
        else:
            print(f"{model_name}: failed with {model_summary['error']}")
    intersections = artifact["intersection_counts"]
    print(
        "Intersections: "
        f"tokenizer pairs {intersections['tokenizer_metric_pair_count']}; "
        f"accepted JSONL pairs {intersections['tokenizer_accepted_pair_count']}; "
        f"animate targets {intersections['animate_target_count']}; "
        f"inanimate targets {intersections['inanimate_target_count']}; "
        f"avg_LD_pairs sign-filtered pairs {intersections['model_success_pair_count']}"
    )


def run_tokenization_check(args: argparse.Namespace) -> None:
    from circuit_finder_core import run_tokenization_safety_check

    diagnostics = run_tokenization_safety_check(
        model_name=args.model,
        start=path_or_none(args.start_path),
        target_source=args.target_source,
    )
    alignment = diagnostics["raw_dataset_alignment"]
    targets = diagnostics["target_sets"]
    print(f"Tokenization safety check for {diagnostics['model_name']}")
    print(f"Target source: {diagnostics['target_source']} ({diagnostics['target_source_path']})")
    if diagnostics["requested_model_name"] != diagnostics["model_name"]:
        print(f"Requested alias: {diagnostics['requested_model_name']}")
    if diagnostics["model_note"]:
        print(f"Note: {diagnostics['model_note']}")
    print(
        "Dataset alignment: "
        f"{alignment['fully_aligned']}/{alignment['checked_pairs']} fully aligned; "
        f"{alignment['sequence_length_mismatch']} sequence length mismatches; "
        f"{alignment['patient_span_misaligned']} patient span mismatches; "
        f"{alignment['verb_span_misaligned']} verb span mismatches; "
        f"{alignment['patient_not_single_token']} patient multi-token cases; "
        f"{alignment['verb_not_single_token']} verb multi-token cases; "
        f"{alignment['metadata_missing']} missing metadata."
    )
    print(
        "Target sets: "
        f"animate {targets['animate']['single_token_count']}/{targets['animate']['total']} single-token; "
        f"inanimate {targets['inanimate']['single_token_count']}/{targets['inanimate']['total']} single-token."
    )
    filtered_targets = diagnostics["filtered_target_sets"]
    print(
        "Filtered target sets: "
        f"animate {filtered_targets['animate']['total']}; "
        f"inanimate {filtered_targets['inanimate']['total']}; "
        f"path {diagnostics['target_filter_path']}."
    )


def run_diagnose_model(args: argparse.Namespace) -> None:
    from circuit_finder_core import run_model_diagnostic

    artifact = run_model_diagnostic(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        filter_batch_size=args.filter_batch_size,
        seed=args.seed,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        target_source=args.target_source,
        output_day=args.output_day,
        output_dir=path_or_none(args.output_dir),
        save=args.save,
        save_details=args.save_details,
        save_debug_details=args.save_debug_details,
        start=path_or_none(args.start_path),
    )
    summary = artifact["dataset_summary"]
    alignment = artifact["tokenization_diagnostics"]["raw_dataset_alignment"]
    targets = artifact["tokenization_diagnostics"]["target_sets"]

    print(f"Model diagnostic for {summary['target_model']}")
    print(f"Target source: {summary['target_source']} ({summary['target_source_path']})")
    if summary["target_model_requested"] != summary["target_model"]:
        print(f"Requested alias: {summary['target_model_requested']}")
    if summary.get("source_success_cache_path"):
        print(
            "Source success cache: "
            f"{summary.get('source_success_cache_status')} "
            f"{summary['source_success_cache_path']}"
        )
    print(
        "Raw task accuracy: "
        f"{summary['target_raw_accuracy']['pair_success']['rate']:.4f} "
        f"({summary['target_raw_accuracy']['pair_success']['count']}/"
        f"{summary['target_raw_accuracy']['example_count']})"
    )
    print(
        "Accuracy on source-success pool: "
        f"{summary['target_on_source_accuracy']['pair_success']['rate']:.4f} "
        f"({summary['target_on_source_accuracy']['pair_success']['count']}/"
        f"{summary['target_on_source_accuracy']['example_count']})"
    )
    print(
        "Filtered examples for experiments: "
        f"{summary['target_filtered_count']} "
        f"using {summary['target_filter_policy']}"
    )
    print(
        "Dataset alignment: "
        f"{alignment['fully_aligned']}/{alignment['checked_pairs']} fully aligned; "
        f"{alignment['sequence_length_mismatch']} sequence length mismatches; "
        f"{alignment['patient_span_misaligned']} patient span mismatches; "
        f"{alignment['verb_span_misaligned']} verb span mismatches; "
        f"{alignment['patient_not_single_token']} patient multi-token cases; "
        f"{alignment['verb_not_single_token']} verb multi-token cases; "
        f"{alignment['metadata_missing']} missing metadata."
    )
    print(
        "Target sets: "
        f"animate {targets['animate']['single_token_count']}/{targets['animate']['total']} single-token; "
        f"inanimate {targets['inanimate']['single_token_count']}/{targets['inanimate']['total']} single-token."
    )
    if "summary" in artifact["paths"]:
        print(f"Saved diagnostic summary to {artifact['paths']['summary']}")


def make_eap_config(args: argparse.Namespace):
    from circuit_finder_core import EAPExperimentConfig

    return EAPExperimentConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        seed=args.seed,
        discovery_sample_size=args.discovery_sample_size,
        discovery_margin_threshold=args.discovery_margin_threshold,
        filter_batch_size=args.filter_batch_size,
        attribution_batch_size=args.attribution_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        ig_steps=args.ig_steps,
        budgets=tuple(args.budgets) if args.budgets is not None else None,
        budget_max_fraction=args.budget_max_fraction,
        budget_floor=args.budget_floor,
        budget_tail_points=args.budget_tail_points,
        budget_early_stop=args.budget_early_stop,
        budget_early_stop_threshold=args.budget_early_stop_threshold,
        budget_early_stop_patience=args.budget_early_stop_patience,
        budget_early_stop_min_delta=args.budget_early_stop_min_delta,
        budget_early_stop_start_budget=args.budget_early_stop_start_budget,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        target_source=args.target_source,
        circuit_finder_day=getattr(args, "circuit_finder_day", None),
        importance_quantile=getattr(args, "importance_quantile", 0.10),
        component_discovery_threshold=getattr(args, "component_threshold", None),
    )


def run_component_discovery(args: argparse.Namespace) -> None:
    from circuit_finder_core import (
        DEFAULT_THRESHOLDS,
        ComponentDiscoveryConfig,
        run_component_discovery_experiment,
    )

    config = ComponentDiscoveryConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        seed=args.seed,
        discovery_sample_size=args.discovery_sample_size,
        discovery_margin_threshold=args.discovery_margin_threshold,
        filter_batch_size=args.filter_batch_size,
        patch_batch_size=args.patch_batch_size,
        thresholds=tuple(args.thresholds) if args.thresholds is not None else DEFAULT_THRESHOLDS,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        target_source=args.target_source,
    )
    artifact = run_component_discovery_experiment(
        config=config,
        start=path_or_none(args.start_path),
    )
    print(f"Saved component discovery outputs to {artifact['paths']['output_dir']}")


def run_concept_extraction(args: argparse.Namespace) -> None:
    from circuit_finder_core import (
        DEFAULT_CONCEPT_ALPHA_GRID,
        DEFAULT_CONCEPT_HOOK_POINTS,
        ConceptExtractionConfig,
        run_concept_extraction_experiment,
    )

    config = ConceptExtractionConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        seed=args.seed,
        filter_batch_size=args.filter_batch_size,
        extraction_batch_size=args.extraction_batch_size,
        steering_batch_size=args.steering_batch_size,
        alpha_grid=tuple(args.alpha_grid) if args.alpha_grid is not None else DEFAULT_CONCEPT_ALPHA_GRID,
        hook_points=tuple(args.hook_points) if args.hook_points is not None else DEFAULT_CONCEPT_HOOK_POINTS,
        normalize_concept_vector=args.normalize_concept_vector,
        selection_effect_fraction=args.selection_effect_fraction,
        random_control_repeats=args.random_control_repeats,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        max_concept_examples=args.max_concept_examples,
        target_filter_policy=args.target_filter_policy,
        target_source=args.target_source,
    )
    artifact = run_concept_extraction_experiment(
        config=config,
        start=path_or_none(args.start_path),
    )
    selected = artifact["selected"]
    test_summary = artifact["test_summary"]
    print(f"Saved concept-extraction outputs to {artifact['paths']['output_dir']}")
    print(
        "Selected "
        f"{selected['hook_name']} alpha={float(selected['alpha']):g}; "
        f"validation effect={float(selected['signed_effect_mean']):.6g}; "
        f"test effect={float(test_summary['signed_effect_mean']):.6g}; "
        f"random mean={float(artifact['random_control_summary']['signed_effect_mean']):.6g}"
    )


def run_eap_full(args: argparse.Namespace) -> None:
    from circuit_finder_core import run_full_model_eap_experiment

    artifact = run_full_model_eap_experiment(
        config=make_eap_config(args),
        start=path_or_none(args.start_path),
    )
    print(f"Saved full-model EAP-IG outputs to {artifact['paths']['output_dir']}")


def run_eap_selected(args: argparse.Namespace) -> None:
    from circuit_finder_core import run_selected_components_eap_experiment

    artifact = run_selected_components_eap_experiment(
        config=make_eap_config(args),
        start=path_or_none(args.start_path),
    )
    print(f"Saved selected-component EAP-IG outputs to {artifact['paths']['output_dir']}")


def run_eap_shadow_rediscovery(args: argparse.Namespace) -> None:
    from circuit_finder_core import EAPShadowRediscoveryConfig, run_shadow_rediscovery_experiment

    config = EAPShadowRediscoveryConfig(
        model_name=args.model,
        shared_filter_model_names=tuple(args.shared_filter_models),
        dataset_set_name=args.dataset_set,
        seed=args.seed,
        discovery_sample_size=args.discovery_sample_size,
        discovery_margin_threshold=args.discovery_margin_threshold,
        filter_batch_size=args.filter_batch_size,
        attribution_batch_size=args.attribution_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        ig_steps=args.ig_steps,
        budgets=tuple(args.budgets) if args.budgets is not None else None,
        budget_max_fraction=args.budget_max_fraction,
        budget_floor=args.budget_floor,
        budget_tail_points=args.budget_tail_points,
        budget_early_stop=args.budget_early_stop,
        budget_early_stop_threshold=args.budget_early_stop_threshold,
        budget_early_stop_patience=args.budget_early_stop_patience,
        budget_early_stop_min_delta=args.budget_early_stop_min_delta,
        budget_early_stop_start_budget=args.budget_early_stop_start_budget,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        target_source=args.target_source,
        main_experiment_path=args.main_experiment_path,
        source_faithfulness_threshold=args.source_faithfulness_threshold,
        variants=tuple(args.variants) if args.variants is not None else None,
        top_edge_count=args.top_edge_count,
        skip_existing=args.skip_existing,
    )
    artifact = run_shadow_rediscovery_experiment(
        config=config,
        start=path_or_none(args.start_path),
    )
    print(f"Saved shadow rediscovery outputs to {artifact['paths']['output_dir']}")


def run_dual_set_eap(args: argparse.Namespace) -> None:
    from circuit_finder_core import DualSetExperimentConfig, run_dual_set_eap_workflow

    config = DualSetExperimentConfig(
        model_name=args.model,
        shared_filter_model_names=tuple(args.shared_filter_models),
        dataset_set_names=tuple(args.dataset_sets),
        seed=args.seed,
        discovery_sample_size=args.discovery_sample_size,
        discovery_margin_threshold=args.discovery_margin_threshold,
        filter_batch_size=args.filter_batch_size,
        attribution_batch_size=args.attribution_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        ig_steps=args.ig_steps,
        budgets=tuple(args.budgets) if args.budgets is not None else None,
        budget_max_fraction=args.budget_max_fraction,
        budget_floor=args.budget_floor,
        budget_tail_points=args.budget_tail_points,
        budget_early_stop=args.budget_early_stop,
        budget_early_stop_threshold=args.budget_early_stop_threshold,
        budget_early_stop_patience=args.budget_early_stop_patience,
        budget_early_stop_min_delta=args.budget_early_stop_min_delta,
        budget_early_stop_start_budget=args.budget_early_stop_start_budget,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        target_source=args.target_source,
        run_diagnose=args.run_diagnose,
        run_eap=args.run_eap,
    )
    artifact = run_dual_set_eap_workflow(
        config=config,
        start=path_or_none(args.start_path),
    )
    for dataset_set_name, run_bundle in artifact["artifacts"].items():
        if "diagnose_model" in run_bundle:
            print(
                f"Saved {dataset_set_name} diagnostic outputs to "
                f"{run_bundle['diagnose_model']['paths']['output_dir']}"
            )
        if "eap_full" in run_bundle:
            print(
                f"Saved {dataset_set_name} EAP-IG outputs to "
                f"{run_bundle['eap_full']['paths']['output_dir']}"
            )


def run_eap_localization(args: argparse.Namespace) -> None:
    from run_eap_localization import LocalizationConfig, run_localization_experiment

    top_k = tuple(args.budgets) if args.budgets is not None else tuple(args.top_k)
    config = LocalizationConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        sample_sizes=tuple(args.sample_sizes),
        seeds=tuple(args.seeds),
        top_k=top_k,
        random_repeats=args.random_repeats,
        discovery_margin_threshold=args.discovery_margin_threshold,
        filter_batch_size=args.filter_batch_size,
        attribution_batch_size=args.attribution_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        ig_steps=args.ig_steps,
        max_validation_examples=args.max_validation_examples,
        budget_max_fraction=args.budget_max_fraction,
        budget_floor=args.budget_floor,
        budget_tail_points=args.budget_tail_points,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        skip_random_baselines=args.skip_random_baselines,
        skip_existing=args.skip_existing,
        import_edge_rankings_path=args.import_edge_rankings_path,
        import_summary_path=args.import_summary_path,
        import_sample_size=args.import_sample_size,
        import_seed=args.import_seed,
    )
    manifest = run_localization_experiment(
        config=config,
        start=path_or_none(args.start_path),
    )
    print(f"Saved EAP-IG localization outputs to {manifest['paths']['output_root']}")
    print(f"Manifest: {manifest['paths']['manifest']}")


def run_eap_conditional_ablation(args: argparse.Namespace) -> None:
    from run_conditional_ablation import (
        ConditionalAblationConfig,
        run_conditional_ablation_experiment,
    )

    config = ConditionalAblationConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        discovery_margin_threshold=args.discovery_margin_threshold,
        filter_batch_size=args.filter_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        max_validation_examples=args.max_validation_examples,
        output_day=args.output_day,
        dataset_filter_path=args.dataset_filter_path,
        refresh_dataset_filter=args.refresh_dataset_filter,
        cache_dataset_filter=args.cache_dataset_filter,
        max_filter_examples=args.max_filter_examples,
        target_filter_policy=args.target_filter_policy,
        dataset_mode=args.dataset_mode,
        dataset_set=args.dataset_set,
        named_entity_discovery_dir=args.named_entity_discovery_dir,
        target_source=args.target_source,
        target_token_mode=args.target_token_mode,
        localization_source_path=args.localization_source_path,
        edge_rankings_path=args.edge_rankings_path,
        sample_size=args.discovery_sample_size,
        seed=args.seed,
        protected_budget=args.protected_budget,
        ablated_budget=args.ablated_budget,
        candidate_start_rank=args.candidate_start_rank,
        candidate_end_rank=args.candidate_end_rank,
        band_size=args.band_size,
        sample_count=args.sample_count,
        sampling_strategy=args.sampling_strategy,
        random_seed=args.random_seed,
    )
    manifest = run_conditional_ablation_experiment(
        config=config,
        start=path_or_none(args.start_path),
    )
    print(f"Saved conditional ablation outputs to {manifest['paths']['output_dir']}")
    print(f"Summary: {manifest['paths']['summary']}")


def run_verb_noise_control(args: argparse.Namespace) -> None:
    from control_runners import VerbNoiseControlConfig, run_verb_noise_control

    config = VerbNoiseControlConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        seed=args.seed,
        filter_batch_size=args.filter_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        target_filter_policy=args.target_filter_policy,
        sigma=args.sigma,
        noise_site=args.noise_site,
        output_day=args.output_day,
        main_experiment_path=args.main_experiment_path,
        max_budgets=args.max_budgets,
        run_second_stage_discovery_on_ambiguous=args.run_second_stage_discovery_on_ambiguous,
    )
    summary = run_verb_noise_control(
        config=config,
        main_artifact_or_rankings=args.main_experiment_path,
        sigma=args.sigma,
        start=path_or_none(args.start_path),
    )
    print(f"Saved verb-noise control outputs to {summary['paths']['output_dir']}")


def run_by_to_near_control(args: argparse.Namespace) -> None:
    from control_runners import PrepositionControlConfig, run_preposition_control

    config = PrepositionControlConfig(
        model_name=args.model,
        dataset_filter_model_name=args.dataset_filter_model,
        seed=args.seed,
        filter_batch_size=args.filter_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        target_filter_policy=args.target_filter_policy,
        replacement_from=args.replacement_from,
        replacement_to=args.replacement_to,
        output_day=args.output_day,
        main_experiment_path=args.main_experiment_path,
        max_budgets=args.max_budgets,
        run_second_stage_discovery_on_ambiguous=args.run_second_stage_discovery_on_ambiguous,
    )
    summary = run_preposition_control(
        config=config,
        main_artifact_or_rankings=args.main_experiment_path,
        start=path_or_none(args.start_path),
    )
    print(f"Saved by->near control outputs to {summary['paths']['output_dir']}")


def run_blimp_passive_prefix_control(args: argparse.Namespace) -> None:
    from control_runners import (
        BlimpPassivePrefixControlConfig,
        run_blimp_passive_prefix_control as run_blimp_prefix_impl,
    )

    config = BlimpPassivePrefixControlConfig(
        model_name=args.model,
        evaluation_batch_size=args.evaluation_batch_size,
        output_day=args.output_day,
        main_experiment_path=args.main_experiment_path,
        source_faithfulness_threshold=args.source_faithfulness_threshold,
        budget=args.budget,
    )
    summary = run_blimp_prefix_impl(
        config=config,
        main_artifact_or_rankings=args.main_experiment_path,
        start=path_or_none(args.start_path),
    )
    print(f"Saved BLiMP passive prefix control outputs to {summary['paths']['output_dir']}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        from run_circuit_comparison import main as compare_main

        compare_main(argv[1:])
        return

    args = make_parser().parse_args(argv)
    if args.experiment == "prepare-tokenization-filters":
        run_prepare_tokenization_filters(args)
    elif args.experiment == "metric-investigation-score":
        run_metric_investigation_score(args)
    elif args.experiment == "prepare-dataset":
        run_prepare_dataset(args)
    elif args.experiment == "tokenization-check":
        run_tokenization_check(args)
    elif args.experiment == "diagnose-model":
        run_diagnose_model(args)
    elif args.experiment == "component-discovery":
        run_component_discovery(args)
    elif args.experiment == "concept-extraction":
        run_concept_extraction(args)
    elif args.experiment == "eap-full":
        run_eap_full(args)
    elif args.experiment == "eap-selected":
        run_eap_selected(args)
    elif args.experiment == "eap-shadow-rediscovery":
        run_eap_shadow_rediscovery(args)
    elif args.experiment == "eap-localization":
        run_eap_localization(args)
    elif args.experiment == "eap-conditional-ablation":
        run_eap_conditional_ablation(args)
    elif args.experiment == "dual-set-eap":
        run_dual_set_eap(args)
    elif args.experiment == "control-verb-noise":
        run_verb_noise_control(args)
    elif args.experiment == "control-by-to-near":
        run_by_to_near_control(args)
    elif args.experiment == "control-blimp-passive-prefix":
        run_blimp_passive_prefix_control(args)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")


if __name__ == "__main__":
    main()

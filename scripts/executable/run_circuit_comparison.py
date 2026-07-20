from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils import (
    canonical_model_name,
    ensure_dir,
    resolve_animacy_circuit_root,
    safe_model_name,
    save_csv,
    save_json,
    timestamp_tag,
)


FULL_MODEL = "full_model"
SELECTED_COMPONENTS = "selected_components"
REQUIRED_COLUMNS = {
    "collapsed_edge_budget",
    "expanded_edge_count",
    "induced_node_count",
    "faithfulness_mean",
}
OPTIONAL_METRICS = ("accuracy_mean", "faithfulness_std", "accuracy_std", "validation_examples")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved full-model and selected-component EAP-IG budget sweeps "
            "without running patching, attribution, or validation experiments."
        )
    )
    parser.add_argument(
        "--day",
        default=None,
        help=(
            "Result day under animacy-circuit/results/eap_ig/<model>/<day>. "
            "Defaults to the latest day with both full-model and selected-component budget sweeps."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model name for results under results/eap_ig/<model-slug>/<day>. "
            "If omitted, the latest model/day pair with both sweeps is discovered automatically."
        ),
    )
    parser.add_argument(
        "--component-day",
        default=None,
        help=(
            "Optional component-discovery day used by the selected-components run. "
            "When provided, the selected-components summary must reference this day."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional explicit path to the animacy-circuit/results directory.",
    )
    parser.add_argument(
        "--full-model-dir",
        default=None,
        help="Optional directory containing full_model_budget_sweep_*.csv.",
    )
    parser.add_argument(
        "--selected-components-dir",
        default=None,
        help="Optional directory containing selected_components_budget_sweep_*.csv.",
    )
    parser.add_argument(
        "--full-model-sweep",
        default=None,
        help="Optional explicit full_model_budget_sweep CSV path.",
    )
    parser.add_argument(
        "--selected-components-sweep",
        default=None,
        help="Optional explicit selected_components_budget_sweep CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where to save comparison CSV/JSON. Implies --save. "
            "Defaults to results/eap_ig/<model>/<day>/comparison when --save is used."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write comparison CSV/JSON artifacts. By default the script only prints the comparison.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the comparison summary without writing comparison artifacts.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for treating mean faithfulness differences as ties.",
    )
    parser.add_argument(
        "--start-path",
        default=None,
        help="Optional path used to locate the animacy-circuit project root.",
    )
    return parser.parse_args(argv)


def latest_path(paths: list[Path], description: str) -> Path:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"Could not find {description}.")
    return max(existing, key=lambda path: path.stat().st_mtime)


def latest_directory(paths: list[Path], description: str) -> Path:
    existing = [path for path in paths if path.is_dir()]
    if not existing:
        raise FileNotFoundError(f"Could not find {description}.")
    return max(existing, key=lambda path: path.stat().st_mtime)


def experiment_day_dir(
    results_root: Path,
    experiment_name: str,
    model_slug: str,
    day: str,
) -> Path:
    return results_root / experiment_name / model_slug / day


def candidate_model_slugs(results_root: Path) -> list[str]:
    model_slugs: set[str] = set()
    eap_root = results_root / "eap_ig"
    if eap_root.is_dir():
        for model_dir in eap_root.iterdir():
            if model_dir.is_dir():
                model_slugs.add(model_dir.name)
    for experiment_name in ("eap_ig_full_model", "eap_ig_selected_components"):
        experiment_dir = results_root / experiment_name
        if experiment_dir.is_dir():
            for model_dir in experiment_dir.iterdir():
                if model_dir.is_dir():
                    model_slugs.add(model_dir.name)
    return sorted(model_slugs)


def available_days_for_experiment(
    results_root: Path,
    experiment_name: str,
    model_slug: str,
) -> set[str]:
    if experiment_name == "eap_ig_full_model":
        experiment_dir = results_root / "eap_ig" / model_slug
        if experiment_dir.is_dir():
            return {
                path.name
                for path in experiment_dir.iterdir()
                if path.is_dir() and (path / "full_model").is_dir()
            }
    if experiment_name == "eap_ig_selected_components":
        experiment_dir = results_root / "eap_ig" / model_slug
        if experiment_dir.is_dir():
            return {
                path.name
                for path in experiment_dir.iterdir()
                if path.is_dir() and (path / "selected_components").is_dir()
            }
    experiment_dir = results_root / experiment_name / model_slug
    if not experiment_dir.is_dir():
        return set()
    return {path.name for path in experiment_dir.iterdir() if path.is_dir()}


def selected_components_component_day(directory: Path) -> str | None:
    summaries = sorted(directory.glob("selected_components_summary_*.json"))
    if not summaries:
        return None
    latest = max(summaries, key=lambda path: path.stat().st_mtime)
    with latest.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    return artifact.get("circuit_finder_day")


def resolve_full_model_dir(args: argparse.Namespace, results_root: Path, model_slug: str, day: str) -> Path:
    if args.full_model_dir:
        return Path(args.full_model_dir).resolve()

    candidates = [
        results_root / "eap_ig" / model_slug / day / "full_model",
        experiment_day_dir(results_root, "eap_ig_full_model", model_slug, day),
        results_root / day / model_slug / "eap_ig_full_model",
        results_root / day / "eap_ig_full_model",
    ]
    return latest_directory(candidates, "full-model result directory")


def resolve_selected_components_dir(
    args: argparse.Namespace,
    results_root: Path,
    model_slug: str,
    day: str,
) -> Path:
    if args.selected_components_dir:
        return Path(args.selected_components_dir).resolve()

    candidates = [
        results_root / "eap_ig" / model_slug / day / "selected_components",
        experiment_day_dir(results_root, "eap_ig_selected_components", model_slug, day),
        results_root / day / model_slug / "eap_ig_selected_components",
        results_root / day / f"eap_ig_selected_components_from_{args.component_day or day}",
        results_root / day / "eap_ig_selected_components",
    ]
    directory = latest_directory(candidates, "selected-components result directory")
    if args.component_day is not None:
        actual_component_day = selected_components_component_day(directory)
        if actual_component_day is not None and actual_component_day != args.component_day:
            raise FileNotFoundError(
                "Selected-components result directory does not match the requested "
                f"component-discovery day {args.component_day!r}: {directory}"
            )
    return directory


def find_budget_sweep(directory: Path, pattern: str) -> Path:
    return latest_path(list(directory.glob(pattern)), f"{pattern} in {directory}")


def discover_result_scope(results_root: Path, requested_model: str | None = None) -> tuple[str, str]:
    if requested_model is not None:
        model_slug = safe_model_name(canonical_model_name(requested_model))
        candidate_days = sorted(
            available_days_for_experiment(results_root, "eap_ig_full_model", model_slug)
            & available_days_for_experiment(results_root, "eap_ig_selected_components", model_slug)
        )
        if not candidate_days:
            raise FileNotFoundError(
                f"No saved full-model and selected-components sweeps found for model {model_slug!r}."
            )
        return model_slug, candidate_days[-1]

    candidates: list[tuple[str, str]] = []
    for model_slug in candidate_model_slugs(results_root):
        candidate_days = sorted(
            available_days_for_experiment(results_root, "eap_ig_full_model", model_slug)
            & available_days_for_experiment(results_root, "eap_ig_selected_components", model_slug)
        )
        candidates.extend((day, model_slug) for day in candidate_days)

    if not candidates:
        raise FileNotFoundError(
            "No saved model/day pair under the new results layout contains both budget sweeps."
        )
    day, model_slug = max(candidates, key=lambda item: item[0])
    return model_slug, day


def discover_model_for_day(results_root: Path, day: str) -> str:
    candidates: list[str] = []
    for model_slug in candidate_model_slugs(results_root):
        available_full = available_days_for_experiment(results_root, "eap_ig_full_model", model_slug)
        available_selected = available_days_for_experiment(
            results_root,
            "eap_ig_selected_components",
            model_slug,
        )
        if day in available_full and day in available_selected:
            candidates.append(model_slug)

    if not candidates:
        raise FileNotFoundError(
            f"No saved model under the new results layout has both budget sweeps for day {day!r}."
        )
    return sorted(candidates)[-1]


def resolve_sweep_paths(args: argparse.Namespace) -> dict[str, Any]:
    project_root = resolve_animacy_circuit_root(
        Path(args.start_path) if args.start_path else None
    )
    results_root = Path(args.results_root).resolve() if args.results_root else project_root / "results"
    if args.model is not None:
        model_slug = safe_model_name(canonical_model_name(args.model))
        if args.day is None:
            _, day = discover_result_scope(results_root, args.model)
        else:
            day = args.day
    elif args.day is not None:
        day = args.day
        model_slug = discover_model_for_day(results_root, day)
    else:
        model_slug, day = discover_result_scope(results_root, None)

    full_model_sweep = (
        Path(args.full_model_sweep).resolve()
        if args.full_model_sweep
        else find_budget_sweep(
            resolve_full_model_dir(args, results_root, model_slug, day),
            "full_model_budget_sweep_*.csv",
        )
    )

    selected_components_sweep = (
        Path(args.selected_components_sweep).resolve()
        if args.selected_components_sweep
        else find_budget_sweep(
            resolve_selected_components_dir(args, results_root, model_slug, day),
            "selected_components_budget_sweep_*.csv",
        )
    )

    return {
        "project_root": project_root,
        "results_root": results_root,
        "model_slug": model_slug,
        "day": day,
        "full_model_sweep": full_model_sweep,
        "selected_components_sweep": selected_components_sweep,
    }


def load_budget_sweep(path: Path, pipeline: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Budget sweep file does not exist: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required budget sweep columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["pipeline"] = pipeline
    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS.union(OPTIONAL_METRICS)
        if column in frame.columns
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    return frame.sort_values("collapsed_edge_budget").reset_index(drop=True)


def winner_from_delta(delta: float, tolerance: float) -> str:
    if delta > tolerance:
        return SELECTED_COMPONENTS
    if delta < -tolerance:
        return FULL_MODEL
    return "tie"


def compare_budget_sweeps(
    full_model: pd.DataFrame,
    selected_components: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    merged = full_model.merge(
        selected_components,
        on="collapsed_edge_budget",
        how="inner",
        suffixes=(f"_{FULL_MODEL}", f"_{SELECTED_COMPONENTS}"),
    )
    if merged.empty:
        raise ValueError("The two budget sweeps have no collapsed-edge budgets in common.")

    for metric in ("faithfulness_mean", "accuracy_mean"):
        full_column = f"{metric}_{FULL_MODEL}"
        selected_column = f"{metric}_{SELECTED_COMPONENTS}"
        if full_column in merged.columns and selected_column in merged.columns:
            merged[f"{metric}_delta"] = merged[selected_column] - merged[full_column]

    for count in ("expanded_edge_count", "induced_node_count"):
        full_column = f"{count}_{FULL_MODEL}"
        selected_column = f"{count}_{SELECTED_COMPONENTS}"
        if full_column in merged.columns and selected_column in merged.columns:
            merged[f"{count}_delta"] = merged[selected_column] - merged[full_column]

    merged["winner_by_faithfulness"] = merged["faithfulness_mean_delta"].apply(
        lambda delta: winner_from_delta(float(delta), tolerance)
    )
    return merged


def summarize_pipeline(
    frame: pd.DataFrame,
    pipeline: str,
    common_budgets: set[int],
) -> dict[str, Any]:
    common = frame[frame["collapsed_edge_budget"].isin(common_budgets)].copy()
    best_idx = common["faithfulness_mean"].idxmax()

    summary: dict[str, Any] = {
        "pipeline": pipeline,
        "common_budget_count": int(len(common)),
        "mean_faithfulness": float(common["faithfulness_mean"].mean()),
        "mean_induced_node_count": float(common["induced_node_count"].mean()),
        "mean_expanded_edge_count": float(common["expanded_edge_count"].mean()),
        "max_faithfulness": float(common.loc[best_idx, "faithfulness_mean"]),
        "best_budget_by_faithfulness": int(common.loc[best_idx, "collapsed_edge_budget"]),
    }

    if "accuracy_mean" in common.columns:
        summary["mean_accuracy"] = float(common["accuracy_mean"].mean())
        best_accuracy_idx = common["accuracy_mean"].idxmax()
        summary["max_accuracy"] = float(common.loc[best_accuracy_idx, "accuracy_mean"])
        summary["best_budget_by_accuracy"] = int(
            common.loc[best_accuracy_idx, "collapsed_edge_budget"]
        )

    return summary


def choose_pipeline(
    full_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    faithfulness_delta = (
        selected_summary["mean_faithfulness"] - full_summary["mean_faithfulness"]
    )
    if abs(faithfulness_delta) > tolerance:
        chosen = SELECTED_COMPONENTS if faithfulness_delta > 0 else FULL_MODEL
        reason = "higher_mean_faithfulness"
    else:
        induced_delta = (
            selected_summary["mean_induced_node_count"]
            - full_summary["mean_induced_node_count"]
        )
        if abs(induced_delta) > tolerance:
            chosen = SELECTED_COMPONENTS if induced_delta < 0 else FULL_MODEL
            reason = "faithfulness_tie_smaller_mean_induced_node_count"
        else:
            chosen = "tie"
            reason = "faithfulness_and_induced_node_tie"

    return {
        "chosen_pipeline": chosen,
        "reason": reason,
        "mean_faithfulness_delta_selected_minus_full": float(faithfulness_delta),
    }


def build_artifact(
    paths: dict[str, Any],
    full_model: pd.DataFrame,
    selected_components: pd.DataFrame,
    comparison: pd.DataFrame,
    tolerance: float,
) -> dict[str, Any]:
    common_budgets = set(comparison["collapsed_edge_budget"].astype(int).tolist())
    full_budgets = set(full_model["collapsed_edge_budget"].astype(int).tolist())
    selected_budgets = set(selected_components["collapsed_edge_budget"].astype(int).tolist())

    full_summary = summarize_pipeline(full_model, FULL_MODEL, common_budgets)
    selected_summary = summarize_pipeline(
        selected_components,
        SELECTED_COMPONENTS,
        common_budgets,
    )

    return {
        "paths": {
            "project_root": paths["project_root"],
            "results_root": paths["results_root"],
            "model_slug": paths["model_slug"],
            "full_model_sweep": paths["full_model_sweep"],
            "selected_components_sweep": paths["selected_components_sweep"],
        },
        "comparison_scope": {
            "model_slug": paths["model_slug"],
            "day": paths["day"],
            "common_budgets": sorted(common_budgets),
            "full_model_only_budgets": sorted(full_budgets - selected_budgets),
            "selected_components_only_budgets": sorted(selected_budgets - full_budgets),
            "tolerance": tolerance,
        },
        "pipeline_summaries": [full_summary, selected_summary],
        "budget_winner_counts": comparison["winner_by_faithfulness"].value_counts().to_dict(),
        "selection_rule": (
            "Compare saved budget sweeps on overlapping collapsed-edge budgets. "
            "Choose higher mean faithfulness; break ties by smaller mean induced node count."
        ),
        "chosen_final_pipeline": choose_pipeline(
            full_summary,
            selected_summary,
            tolerance,
        ),
    }


def save_comparison(
    args: argparse.Namespace,
    paths: dict[str, Any],
    comparison: pd.DataFrame,
    artifact: dict[str, Any],
) -> dict[str, Path]:
    output_dir = ensure_dir(
        Path(args.output_dir).resolve()
        if args.output_dir
        else paths["results_root"] / "eap_ig" / paths["model_slug"] / paths["day"] / "comparison"
    )
    tag = timestamp_tag()
    comparison_path = output_dir / f"circuit_comparison_budget_sweep_{tag}.csv"
    summary_path = output_dir / f"circuit_comparison_summary_{tag}.json"
    save_csv(comparison, comparison_path, index=False)
    save_json(summary_path, artifact)
    return {"comparison_csv": comparison_path, "summary_json": summary_path}


def print_summary(
    artifact: dict[str, Any],
    saved_paths: dict[str, Path] | None,
) -> None:
    summaries = {
        summary["pipeline"]: summary
        for summary in artifact["pipeline_summaries"]
    }
    full = summaries[FULL_MODEL]
    selected = summaries[SELECTED_COMPONENTS]
    chosen = artifact["chosen_final_pipeline"]

    print(
        "Compared saved results for "
        f"{artifact['comparison_scope']['model_slug']} on {artifact['comparison_scope']['day']}"
    )
    print(f"Full-model sweep: {artifact['paths']['full_model_sweep']}")
    print(f"Selected-components sweep: {artifact['paths']['selected_components_sweep']}")
    print(f"Common budgets: {len(artifact['comparison_scope']['common_budgets'])}")
    print(
        "Mean faithfulness: "
        f"{FULL_MODEL}={full['mean_faithfulness']:.6f}, "
        f"{SELECTED_COMPONENTS}={selected['mean_faithfulness']:.6f}, "
        f"delta={chosen['mean_faithfulness_delta_selected_minus_full']:.6f}"
    )
    if "mean_accuracy" in full and "mean_accuracy" in selected:
        print(
            "Mean accuracy: "
            f"{FULL_MODEL}={full['mean_accuracy']:.6f}, "
            f"{SELECTED_COMPONENTS}={selected['mean_accuracy']:.6f}"
        )
    print(f"Chosen final pipeline: {chosen['chosen_pipeline']} ({chosen['reason']})")

    if saved_paths is not None:
        print(f"Saved comparison CSV to {saved_paths['comparison_csv']}")
        print(f"Saved summary JSON to {saved_paths['summary_json']}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_sweep_paths(args)
    full_model = load_budget_sweep(paths["full_model_sweep"], FULL_MODEL)
    selected_components = load_budget_sweep(
        paths["selected_components_sweep"],
        SELECTED_COMPONENTS,
    )
    comparison = compare_budget_sweeps(
        full_model,
        selected_components,
        tolerance=args.tolerance,
    )
    artifact = build_artifact(
        paths=paths,
        full_model=full_model,
        selected_components=selected_components,
        comparison=comparison,
        tolerance=args.tolerance,
    )
    should_save = (args.save or args.output_dir is not None) and not args.no_save
    saved_paths = save_comparison(args, paths, comparison, artifact) if should_save else None
    print_summary(artifact, saved_paths)


if __name__ == "__main__":
    main()

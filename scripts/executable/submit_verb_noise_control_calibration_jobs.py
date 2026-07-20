from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_MODEL_NAMES = (
    "gpt2",
    "meta-llama/Llama-3.2-3B",
    "google/gemma-3-4b-pt",
    "Qwen/Qwen3-4B",
)
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
        description="Submit one sbatch verb-noise calibration job per project model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_NAMES),
    )
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
        help="Shared results day tag. Defaults to today's tag at submission time.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from circuit_finder_core import (
        canonical_model_name,
        date_tag,
        ensure_dir,
        resolve_animacy_circuit_root,
        safe_model_name,
    )

    project_root = resolve_animacy_circuit_root(Path.cwd())
    output_day = args.output_day or date_tag()
    sbatch_script = Path(__file__).with_name("run_verb_noise_control_calibration.sbatch")
    log_dir = ensure_dir(project_root / "results" / output_day / "job_logs" / "verb_noise_calibration")

    print(f"Submitting verb-noise calibration jobs for day {output_day}")
    print(f"Notebook loader should use OUTPUT_DAY = {output_day!r}")

    for model_name in args.models:
        resolved_model_name = canonical_model_name(model_name)
        log_path = log_dir / f"{safe_model_name(resolved_model_name)}-%j.out"
        command = [
            "sbatch",
            "--output",
            str(log_path),
            str(sbatch_script),
            "--model",
            resolved_model_name,
            "--dataset-filter-model",
            args.dataset_filter_model,
            "--seed",
            str(args.seed),
            "--filter-batch-size",
            str(args.filter_batch_size),
            "--target-filter-policy",
            args.target_filter_policy,
            "--noise-site",
            args.noise_site,
            "--output-day",
            output_day,
            "--sigma-multipliers",
            *[str(value) for value in args.sigma_multipliers],
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = completed.stdout.strip()
        print(f"{resolved_model_name}: {stdout}")


if __name__ == "__main__":
    main()

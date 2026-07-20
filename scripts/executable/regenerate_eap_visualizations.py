from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from circuit_finder_core import (
    MODEL_SPECIFIC_CORRECT,
    eap_ig_dataset_set_full_model_dir,
    eap_ig_selected_components_dir,
    save_eap_visualizations,
)
from circuit_finder_paths import resolve_animacy_circuit_root
from utils import canonical_model_name, save_json


PIPELINE_STEMS = {
    "full_model": "full_model",
    "selected_components": "selected_components",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate EAP visualization HTML/PNG artifacts from saved ranking CSVs."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--day", required=True, help="Run day used in artifact filenames, e.g. 2026-05-30.")
    parser.add_argument(
        "--pipeline",
        choices=tuple(PIPELINE_STEMS),
        default="full_model",
        help="Which saved EAP artifact set to render.",
    )
    parser.add_argument(
        "--dataset-set",
        default=MODEL_SPECIFIC_CORRECT,
        help="Dataset-set directory for full-model dual-set EAP outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit directory containing the saved EAP CSVs. Overrides model/day/pipeline defaults.",
    )
    parser.add_argument("--top-k-edges", type=int, default=None)
    parser.add_argument(
        "--static-images",
        action="store_true",
        help="Also export PNGs. By default only interactive HTML is written.",
    )
    parser.add_argument(
        "--start-path",
        type=Path,
        default=None,
        help="Path used to locate the animacy-circuit project root.",
    )
    return parser.parse_args()


def default_output_dir(
    project_root: Path,
    model_name: str,
    day: str,
    pipeline: str,
    dataset_set: str,
) -> Path:
    if pipeline == "full_model":
        return eap_ig_dataset_set_full_model_dir(project_root, model_name, dataset_set, day)
    return eap_ig_selected_components_dir(project_root, model_name, day)


def require_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


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

    edge_frame = require_csv(output_dir / f"{stem}_edges_{args.day}.csv")
    node_frame = require_csv(output_dir / f"{stem}_nodes_{args.day}.csv")
    budget_frame = require_csv(output_dir / f"{stem}_budget_sweep_{args.day}.csv")

    artifacts = save_eap_visualizations(
        project_root=project_root,
        output_dir=output_dir,
        edge_frame=edge_frame,
        node_frame=node_frame,
        budget_frame=budget_frame,
        day=args.day,
        top_k_edges=args.top_k_edges,
        export_static_images=args.static_images,
    )

    artifact_path = output_dir / f"{stem}_visualizations_{args.day}.json"
    save_json(artifact_path, artifacts)
    print(f"Wrote visualization artifact manifest: {artifact_path}")
    for group_name, paths in artifacts.items():
        for kind, path in paths.items():
            print(f"{group_name}.{kind}: {path}")


if __name__ == "__main__":
    main()

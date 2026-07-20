from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_KEYS = ("uid", "clean", "corrupt", "clean_p_yes", "corrupt_p_yes")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths() -> dict[str, Path]:
    base = repo_root() / "dataset" / "semantic_meaningful"
    plot_dir = base / "prefix_score_plots"
    return {
        "merged": base / "scored_semantic_pairs.jsonl",
        "filtered": base / "filtered_pairs_prefix_local_retuned.jsonl",
        "rejected": base / "rejected_pairs_prefix_local_retuned.jsonl",
        "analysis": base / "prefix_score_analysis_retuned.json",
        "plots": plot_dir,
        "plot_clean": plot_dir / "clean_p_yes_distribution.png",
        "plot_corrupt": plot_dir / "corrupt_p_yes_distribution.png",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Analyze clean/corrupt prefix score distributions and regenerate "
            "filtered/rejected outputs with dynamic thresholds."
        )
    )
    parser.add_argument("--merged-path", type=Path, default=paths["merged"])
    parser.add_argument("--filtered-out", type=Path, default=paths["filtered"])
    parser.add_argument("--rejected-out", type=Path, default=paths["rejected"])
    parser.add_argument("--analysis-json", type=Path, default=paths["analysis"])
    parser.add_argument("--plot-clean", type=Path, default=paths["plot_clean"])
    parser.add_argument("--plot-corrupt", type=Path, default=paths["plot_corrupt"])
    parser.add_argument("--clean-min", type=float, default=0.70)
    parser.add_argument("--corrupt-min", type=float, default=0.70)
    parser.add_argument("--curve-points", type=int, default=400)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.0,
        help="KDE bandwidth. Use <=0 to auto-select via Silverman's rule.",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail on malformed rows (default).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip malformed rows instead of failing.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_vals[low])
    frac = pos - low
    return float(sorted_vals[low] + (sorted_vals[high] - sorted_vals[low]) * frac)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q05": None,
            "q25": None,
            "q75": None,
            "q95": None,
        }

    sorted_vals = sorted(values)
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "q05": quantile(sorted_vals, 0.05),
        "q25": quantile(sorted_vals, 0.25),
        "q75": quantile(sorted_vals, 0.75),
        "q95": quantile(sorted_vals, 0.95),
    }


def _silverman_bandwidth(values: list[float]) -> float:
    if len(values) < 2:
        return 0.05
    std = statistics.pstdev(values)
    if std <= 0.0:
        return 0.05
    return max(1.06 * std * (len(values) ** (-0.2)), 0.01)


def _count_curve_bin_width(values: list[float], points: int) -> float:
    if len(values) < 2:
        return max(1.0 / max(points - 1, 1), 0.05)

    sorted_vals = sorted(values)
    q25 = quantile(sorted_vals, 0.25)
    q75 = quantile(sorted_vals, 0.75)
    if q25 is not None and q75 is not None:
        iqr = q75 - q25
        if iqr > 0.0:
            return max(2.0 * iqr * (len(values) ** (-1.0 / 3.0)), 1.0 / max(points - 1, 1))

    std = statistics.pstdev(values)
    if std > 0.0:
        return max(3.49 * std * (len(values) ** (-1.0 / 3.0)), 1.0 / max(points - 1, 1))

    return max(1.0 / max(points - 1, 1), 0.05)


def _kde_curve(values: list[float], points: int, bandwidth: float) -> tuple[list[float], list[float]]:
    x_values = [i / (points - 1) for i in range(points)]
    normalizer = 1.0 / (math.sqrt(2.0 * math.pi) * bandwidth * len(values))
    y_values: list[float] = []
    for x in x_values:
        kernel_sum = 0.0
        for v in values:
            z = (x - v) / bandwidth
            kernel_sum += math.exp(-0.5 * z * z)
        y_values.append(normalizer * kernel_sum)
    return x_values, y_values


def _smoothed_count_curve(values: list[float], points: int, bandwidth: float) -> tuple[list[float], list[float]]:
    x_values, density_values = _kde_curve(values, points=points, bandwidth=bandwidth)
    count_bin_width = _count_curve_bin_width(values, points=points)
    # Scale the KDE into expected counts for a representative histogram-width window.
    y_values = [density * len(values) * count_bin_width for density in density_values]
    return x_values, y_values


def save_smoothed_count_plot(
    values: list[float],
    title: str,
    xlabel: str,
    path: Path,
    curve_points: int,
    bandwidth: float,
) -> None:
    selected_bandwidth = bandwidth if bandwidth > 0.0 else _silverman_bandwidth(values)
    x_values, y_values = _smoothed_count_curve(values, points=curve_points, bandwidth=selected_bandwidth)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, y_values, color="#1f77b4", linewidth=2.2)
    ax.fill_between(x_values, y_values, color="#1f77b4", alpha=0.20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Smoothed count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def normalize_scored_row(row: dict) -> dict:
    missing = [key for key in REQUIRED_KEYS if key not in row]
    if missing:
        raise ValueError(f"Missing required keys {missing}")

    normalized = dict(row)
    normalized["uid"] = str(row["uid"])
    normalized["clean_p_yes"] = float(row["clean_p_yes"])
    normalized["corrupt_p_yes"] = float(row["corrupt_p_yes"])
    return normalized


def main() -> None:
    args = parse_args()
    assert 0.0 <= args.clean_min <= 1.0, "clean-min must be in [0, 1]"
    assert 0.0 <= args.corrupt_min <= 1.0, "corrupt-min must be in [0, 1]"
    assert args.curve_points >= 50, "curve-points must be >= 50"

    if not args.merged_path.exists():
        raise FileNotFoundError(f"Merged scored file not found: {args.merged_path}")

    rows = load_jsonl(args.merged_path)
    if not rows:
        raise ValueError(f"Merged scored file is empty: {args.merged_path}")

    parsed_rows: list[dict] = []
    malformed_rows = 0
    malformed_messages: list[str] = []

    for idx, row in enumerate(rows, start=1):
        try:
            parsed_rows.append(normalize_scored_row(row))
        except Exception as exc:  # pragma: no cover - defensive input handling
            malformed_rows += 1
            if len(malformed_messages) < 10:
                malformed_messages.append(f"row {idx}: {exc}")
            if args.strict:
                continue
            continue

    if args.strict and malformed_rows > 0:
        raise ValueError(
            f"Strict mode validation failed: malformed_rows={malformed_rows} "
            f"| malformed examples: {'; '.join(malformed_messages)}"
        )

    if not parsed_rows:
        raise ValueError("No valid scored rows available after parsing.")

    clean_values = [float(r["clean_p_yes"]) for r in parsed_rows]
    corrupt_values = [float(r["corrupt_p_yes"]) for r in parsed_rows]

    save_smoothed_count_plot(
        clean_values,
        title="Distribution of clean_p_yes",
        xlabel="clean_p_yes",
        path=args.plot_clean,
        curve_points=args.curve_points,
        bandwidth=args.bandwidth,
    )
    save_smoothed_count_plot(
        corrupt_values,
        title="Distribution of corrupt_p_yes",
        xlabel="corrupt_p_yes",
        path=args.plot_corrupt,
        curve_points=args.curve_points,
        bandwidth=args.bandwidth,
    )

    filtered_rows: list[dict] = []
    rejected_rows: list[dict] = []
    rejection_counts = {"clean_low": 0, "corrupt_low": 0, "both_low": 0}

    for row in parsed_rows:
        clean_ok = row["clean_p_yes"] >= args.clean_min
        corrupt_ok = row["corrupt_p_yes"] >= args.corrupt_min

        out_row = dict(row)
        out_row.pop("rejection_reasons", None)

        if clean_ok and corrupt_ok:
            filtered_rows.append(out_row)
            continue

        reasons: list[str] = []
        if not clean_ok:
            reasons.append("clean_low")
        if not corrupt_ok:
            reasons.append("corrupt_low")

        if len(reasons) == 2:
            rejection_counts["both_low"] += 1
        elif reasons:
            rejection_counts[reasons[0]] += 1

        out_row["rejection_reasons"] = reasons
        rejected_rows.append(out_row)

    filtered_written = write_jsonl(args.filtered_out, filtered_rows)
    rejected_written = write_jsonl(args.rejected_out, rejected_rows)

    analysis = {
        "meta": {
            "script": "analyze_and_resplit_prefix_scores.py",
            "description": "Analyze score distributions and regenerate split with runtime thresholds.",
            "clean_min": float(args.clean_min),
            "corrupt_min": float(args.corrupt_min),
            "curve_points": int(args.curve_points),
            "bandwidth": float(args.bandwidth),
            "strict": bool(args.strict),
        },
        "counts": {
            "input_rows": len(rows),
            "valid_rows": len(parsed_rows),
            "malformed_rows": malformed_rows,
            "filtered_rows": filtered_written,
            "rejected_rows": rejected_written,
            "acceptance_rate_over_valid": (filtered_written / len(parsed_rows)) if parsed_rows else None,
            "rejection_reasons": rejection_counts,
        },
        "stats": {
            "clean_p_yes": summarize(clean_values),
            "corrupt_p_yes": summarize(corrupt_values),
        },
        "paths": {
            "input_merged": str(args.merged_path),
            "filtered_out": str(args.filtered_out),
            "rejected_out": str(args.rejected_out),
            "analysis_json": str(args.analysis_json),
            "plot_clean": str(args.plot_clean),
            "plot_corrupt": str(args.plot_corrupt),
        },
        "samples": {
            "malformed_messages": malformed_messages,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.analysis_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.analysis_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"Rows read: {len(rows):,}")
    print(f"Valid rows: {len(parsed_rows):,}")
    print(f"Malformed rows skipped: {malformed_rows:,}")
    print(f"Filtered rows written: {filtered_written:,}")
    print(f"Rejected rows written: {rejected_written:,}")
    print(f"Clean plot: {args.plot_clean.resolve()}")
    print(f"Corrupt plot: {args.plot_corrupt.resolve()}")
    print(f"Analysis JSON: {args.analysis_json.resolve()}")


if __name__ == "__main__":
    main()

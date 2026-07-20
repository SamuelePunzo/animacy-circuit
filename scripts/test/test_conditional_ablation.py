from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True


EXECUTABLE_DIR = Path(__file__).resolve().parents[1] / "executable"
if str(EXECUTABLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTABLE_DIR))

if TORCH_AVAILABLE:
    from run_conditional_ablation import (  # noqa: E402
        build_rank_band_edge_sets,
        filter_candidate_edges,
        resolve_localization_summary_path,
        sample_edge_sets,
    )


def make_edge(rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "collapsed_edge": f"edge_{rank}",
        "parent": f"p{rank}",
        "child": f"c{rank}",
        "abs_score": float(100 - rank),
        "signed_sum": float(100 - rank),
        "underlying_edges": [f"edge_{rank}<0>"],
        "underlying_edge_count": 1,
    }


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for conditional ablation tests")
class ConditionalAblationTests(unittest.TestCase):
    def test_filter_candidate_edges_excludes_protected_and_respects_rank_range(self) -> None:
        ranked_edges = [make_edge(rank) for rank in range(1, 8)]

        candidates = filter_candidate_edges(
            ranked_edges,
            protected_budget=2,
            candidate_start_rank=2,
            candidate_end_rank=6,
        )

        self.assertEqual([edge["rank"] for edge in candidates], [3, 4, 5, 6])

    def test_build_rank_band_edge_sets_uses_non_overlapping_windows(self) -> None:
        candidate_edges = [make_edge(rank) for rank in range(21, 31)]

        sets = build_rank_band_edge_sets(candidate_edges, set_size=5)

        self.assertEqual([edge_set["set_id"] for edge_set in sets], ["rank_21_25", "rank_26_30"])
        self.assertEqual([edge["rank"] for edge in sets[0]["edges"]], [21, 22, 23, 24, 25])
        self.assertEqual([edge["rank"] for edge in sets[1]["edges"]], [26, 27, 28, 29, 30])

    def test_sample_edge_sets_returns_unique_sets_of_requested_size(self) -> None:
        candidate_edges = [make_edge(rank) for rank in range(21, 29)]

        sets = sample_edge_sets(
            candidate_edges,
            set_size=3,
            sample_count=4,
            strategy="uniform",
            random_seed=7,
        )

        self.assertEqual(len(sets), 4)
        seen = set()
        for edge_set in sets:
            ranks = tuple(edge["rank"] for edge in edge_set["edges"])
            self.assertEqual(len(ranks), 3)
            self.assertEqual(len(set(ranks)), 3)
            self.assertNotIn(ranks, seen)
            seen.add(ranks)

    def test_resolve_localization_summary_path_accepts_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "animacy-circuit"
            summary_dir = project_root / "results" / "eap_ig_localization" / "gpt2" / "run_a" / "sample_500" / "seed_42"
            summary_dir.mkdir(parents=True)
            summary_path = summary_dir / "localization_summary_sample_500_seed_42.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "sample_size": 500,
                        "seed": 42,
                        "paths": {"edge_rankings": "animacy-circuit/results/example.csv"},
                    }
                ),
                encoding="utf-8",
            )

            manifest_path = project_root / "results" / "eap_ig_localization" / "gpt2" / "run_a" / "localization_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "sample_size": 500,
                                "seed": 42,
                                "summary": "animacy-circuit/results/eap_ig_localization/gpt2/run_a/sample_500/seed_42/localization_summary_sample_500_seed_42.json",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            resolved = resolve_localization_summary_path(project_root, manifest_path, 500, 42)

            self.assertEqual(resolved, summary_path)


if __name__ == "__main__":
    unittest.main()

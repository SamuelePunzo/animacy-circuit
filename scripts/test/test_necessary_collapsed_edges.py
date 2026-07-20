from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


EXECUTABLE_DIR = Path(__file__).resolve().parents[1] / "executable"
if str(EXECUTABLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTABLE_DIR))

from select_necessary_collapsed_edges import (  # noqa: E402
    first_budget_below_faithfulness,
    find_run_slots,
    ranked_ablation_rows,
    run_analysis,
    split_underlying_edges,
)


class NecessaryCollapsedEdgeTests(unittest.TestCase):
    def test_first_budget_below_faithfulness_selects_smallest_budget(self):
        frame = pd.DataFrame(
            [
                {"mode": "ablate_top", "baseline": "eap_ranked", "matched_random": False, "repeat": 0, "collapsed_edge_budget": 20, "faithfulness_mean": 0.05},
                {"mode": "ablate_top", "baseline": "eap_ranked", "matched_random": False, "repeat": 0, "collapsed_edge_budget": 5, "faithfulness_mean": 0.30},
                {"mode": "ablate_top", "baseline": "eap_ranked", "matched_random": False, "repeat": 0, "collapsed_edge_budget": 10, "faithfulness_mean": 0.09},
            ]
        )

        row = first_budget_below_faithfulness(frame, 0.1)

        self.assertIsNotNone(row)
        self.assertEqual(row["collapsed_edge_budget"], 10)
        self.assertAlmostEqual(row["faithfulness_mean"], 0.09)

    def test_first_budget_below_faithfulness_is_strict(self):
        frame = pd.DataFrame(
            [
                {"mode": "ablate_top", "collapsed_edge_budget": 10, "faithfulness_mean": 0.10},
                {"mode": "ablate_top", "collapsed_edge_budget": 20, "faithfulness_mean": 0.09},
            ]
        )

        row = first_budget_below_faithfulness(frame, 0.1)

        self.assertIsNotNone(row)
        self.assertEqual(row["collapsed_edge_budget"], 20)

    def test_first_budget_below_faithfulness_can_restrict_candidate_budgets(self):
        frame = pd.DataFrame(
            [
                {"mode": "ablate_top", "collapsed_edge_budget": 10, "faithfulness_mean": 0.01},
                {"mode": "ablate_top", "collapsed_edge_budget": 20, "faithfulness_mean": 0.20},
                {"mode": "ablate_top", "collapsed_edge_budget": 50, "faithfulness_mean": 0.05},
            ]
        )

        row = first_budget_below_faithfulness(frame, 0.1, {20, 50})

        self.assertIsNotNone(row)
        self.assertEqual(row["collapsed_edge_budget"], 50)

    def test_ranked_ablation_rows_excludes_random_and_non_ablation_rows(self):
        frame = pd.DataFrame(
            [
                {"mode": "keep_top", "baseline": "eap_ranked", "matched_random": False, "repeat": 0, "collapsed_edge_budget": 1, "faithfulness_mean": 1.0},
                {"mode": "ablate_top", "baseline": "layer_type_matched_random", "matched_random": True, "repeat": 0, "collapsed_edge_budget": 1, "faithfulness_mean": 0.0},
                {"mode": "ablate_top", "baseline": "eap_ranked", "matched_random": False, "repeat": 1, "collapsed_edge_budget": 1, "faithfulness_mean": 0.0},
                {"mode": "ablate_top", "baseline": "eap_ranked", "matched_random": False, "repeat": 0, "collapsed_edge_budget": 5, "faithfulness_mean": 0.5},
            ]
        )

        rows = ranked_ablation_rows(frame)

        self.assertEqual(rows["collapsed_edge_budget"].tolist(), [5])
        self.assertEqual(rows["faithfulness_mean"].tolist(), [0.5])

    def test_split_underlying_edges_handles_pipe_delimited_values(self):
        self.assertEqual(split_underlying_edges("a|b|c"), ["a", "b", "c"])
        self.assertEqual(split_underlying_edges(""), [])
        self.assertEqual(split_underlying_edges(["x", "y"]), ["x", "y"])

    def test_run_analysis_writes_summary_collapsed_and_underlying_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results" / "eap_ig_localization"
            slot_dir = root / "gpt2" / "example_run" / "sample_500" / "seed_42"
            slot_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "collapsed_edge": "input->m0",
                        "parent": "input",
                        "child": "m0",
                        "signed_sum": 0.3,
                        "abs_score": 0.4,
                        "underlying_edges": "input->m0<0>|input->m0<1>",
                        "underlying_edge_count": 2,
                    },
                    {
                        "rank": 2,
                        "collapsed_edge": "m0->logits",
                        "parent": "m0",
                        "child": "logits",
                        "signed_sum": 0.1,
                        "abs_score": 0.2,
                        "underlying_edges": "m0->logits",
                        "underlying_edge_count": 1,
                    },
                ]
            ).to_csv(slot_dir / "edges_sample_500_seed_42.csv", index=False)
            pd.DataFrame(
                [
                    {"mode": "ablate_top", "baseline": "eap_ranked", "matched_random": False, "repeat": 0, "collapsed_edge_budget": 1, "faithfulness_mean": 0.2, "accuracy_mean": 0.7},
                    {"mode": "ablate_top", "baseline": "eap_ranked", "matched_random": False, "repeat": 0, "collapsed_edge_budget": 2, "faithfulness_mean": 0.05, "accuracy_mean": 0.4},
                ]
            ).to_csv(slot_dir / "topk_evaluations_sample_500_seed_42.csv", index=False)

            output_dir = Path(tmpdir) / "out"
            manifest = run_analysis(results_root=root, output_dir=output_dir, threshold=0.1)

            summary = pd.read_csv(manifest["paths"]["summary"])
            collapsed = pd.read_csv(manifest["paths"]["collapsed_edges"])
            underlying = pd.read_csv(manifest["paths"]["underlying_edges"])

            self.assertEqual(summary.loc[0, "status"], "selected")
            self.assertEqual(int(summary.loc[0, "selected_budget"]), 2)
            self.assertEqual(len(collapsed), 2)
            self.assertEqual(len(underlying), 3)
            self.assertEqual(set(underlying["underlying_edge"]), {"input->m0<0>", "input->m0<1>", "m0->logits"})

    def test_main_original_only_filters_probe_smoke_and_named_entity_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results" / "eap_ig_localization"
            run_names = [
                "gpt2_seed_stability",
                "gpt2_seed_stability_probe",
                "import_smoke_2026_05_31",
                "named_entity_truncated_localization_2026-06-16",
            ]
            for run_name in run_names:
                slot_dir = root / "gpt2" / run_name / "sample_500" / "seed_42"
                slot_dir.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            "rank": 1,
                            "collapsed_edge": "input->m0",
                            "parent": "input",
                            "child": "m0",
                            "abs_score": 1.0,
                            "underlying_edges": "input->m0",
                        }
                    ]
                ).to_csv(slot_dir / "edges_sample_500_seed_42.csv", index=False)

            slots = find_run_slots(root, main_original_only=True)

            self.assertEqual(len(slots), 1)
            self.assertEqual(slots[0].run_name, "gpt2_seed_stability")


if __name__ == "__main__":
    unittest.main()

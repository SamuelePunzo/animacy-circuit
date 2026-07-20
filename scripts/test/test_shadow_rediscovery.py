from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


EXECUTABLE_DIR = Path(__file__).resolve().parents[1] / "executable"
if str(EXECUTABLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTABLE_DIR))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_utils_data_stub = types.ModuleType("torch.utils.data")

    class _Dataset:
        pass

    class _DataLoader:
        pass

    torch_utils_data_stub.Dataset = _Dataset
    torch_utils_data_stub.DataLoader = _DataLoader
    torch_utils_stub.data = torch_utils_data_stub
    torch_stub.utils = torch_utils_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_utils_data_stub

try:
    from tqdm.auto import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_auto_stub = types.ModuleType("tqdm.auto")

    def _tqdm(iterable=None, *args, **kwargs):
        del args, kwargs
        return iterable if iterable is not None else []

    tqdm_auto_stub.tqdm = _tqdm
    tqdm_stub.auto = tqdm_auto_stub
    sys.modules["tqdm"] = tqdm_stub
    sys.modules["tqdm.auto"] = tqdm_auto_stub

try:
    from circuit_finder_core import (
        edge_overlap_summary,
        first_budget_reaching_faithfulness,
        resolve_shadow_source_artifacts,
        select_top_edge_groups,
        underlying_edge_name_set,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    edge_overlap_summary = None
    first_budget_reaching_faithfulness = None
    resolve_shadow_source_artifacts = None
    select_top_edge_groups = None
    underlying_edge_name_set = None
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"shadow rediscovery imports unavailable: {IMPORT_ERROR!r}")
class ShadowRediscoveryTests(unittest.TestCase):
    def test_first_budget_reaching_faithfulness_selects_smallest_budget(self):
        frame = pd.DataFrame(
            [
                {"collapsed_edge_budget": 300, "faithfulness_mean": 0.80},
                {"collapsed_edge_budget": 100, "faithfulness_mean": 0.60},
                {"collapsed_edge_budget": 200, "faithfulness_mean": 0.86},
            ]
        )

        row = first_budget_reaching_faithfulness(frame, 0.85)

        self.assertEqual(row["collapsed_edge_budget"], 200)
        self.assertAlmostEqual(row["faithfulness_mean"], 0.86)

    def test_first_budget_reaching_faithfulness_rejects_missing_threshold(self):
        frame = pd.DataFrame(
            [
                {"collapsed_edge_budget": 100, "faithfulness_mean": 0.50},
                {"collapsed_edge_budget": 200, "faithfulness_mean": 0.70},
            ]
        )

        with self.assertRaisesRegex(ValueError, "No budget reaches faithfulness"):
            first_budget_reaching_faithfulness(frame, 0.85)

    def test_select_top_edge_groups_and_underlying_names(self):
        ranked_edges = [
            {"collapsed_edge": "a->b", "underlying_edges": ["a->b<0>", "a->b<1>"]},
            {"collapsed_edge": "b->c", "underlying_edges": ["b->c"]},
            {"collapsed_edge": "c->d", "underlying_edges": ["c->d"]},
        ]

        selected = select_top_edge_groups(ranked_edges, 2)

        self.assertEqual([edge["collapsed_edge"] for edge in selected], ["a->b", "b->c"])
        self.assertEqual(underlying_edge_name_set(selected), {"a->b<0>", "a->b<1>", "b->c"})

    def test_edge_overlap_summary_counts_removed_edges_in_rediscovery(self):
        source_edges = [
            {"collapsed_edge": "a", "underlying_edges": ["a"]},
            {"collapsed_edge": "b", "underlying_edges": ["b"]},
            {"collapsed_edge": "c", "underlying_edges": ["c"]},
        ]
        removed_edges = source_edges[:2]
        rediscovered_edges = [
            {"collapsed_edge": "b", "underlying_edges": ["b"]},
            {"collapsed_edge": "x", "underlying_edges": ["x"]},
        ]

        summary = edge_overlap_summary(rediscovered_edges, source_edges, removed_edges, top_k_values=(2,))

        self.assertEqual(summary["removed_edges_rediscovered_count"], 1)
        self.assertEqual(summary["top_2_source_overlap_count"], 1)
        self.assertEqual(summary["top_2_removed_overlap_count"], 1)

    def test_resolve_shadow_source_artifacts_accepts_explicit_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "full_model"
            source_dir.mkdir()
            edge_path = source_dir / "full_model_edges_2026-06-12.csv"
            budget_path = source_dir / "full_model_budget_sweep_2026-06-12.csv"
            edge_path.write_text("collapsed_edge,parent,child,abs_score,underlying_edges\n", encoding="utf-8")
            budget_path.write_text("collapsed_edge_budget,faithfulness_mean\n", encoding="utf-8")

            paths = resolve_shadow_source_artifacts(
                project_root=Path(tmpdir),
                model_name="gpt2",
                dataset_set_name="model_specific_correct",
                main_experiment_path=source_dir,
            )

            self.assertEqual(paths["source_dir"], source_dir)
            self.assertEqual(paths["edge_path"], edge_path)
            self.assertEqual(paths["budget_path"], budget_path)
            self.assertIsNone(paths["summary_path"])


if __name__ == "__main__":
    unittest.main()

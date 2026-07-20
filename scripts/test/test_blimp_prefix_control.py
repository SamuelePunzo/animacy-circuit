from __future__ import annotations

import sys
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
    from control_runners import (
        prepare_blimp_passive_prefix_rows,
        summarize_blimp_passive_prefix_control,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    prepare_blimp_passive_prefix_rows = None
    summarize_blimp_passive_prefix_control = None
    IMPORT_ERROR = exc


class _Encoded:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        mapping = {
            "Amanda was respected by some": [1, 2, 3, 4, 5],
        }
        if text in mapping:
            return _Encoded(mapping[text])
        return _Encoded(list(range(max(len(text.split()), 1))))


@unittest.skipIf(IMPORT_ERROR is not None, f"control runner imports unavailable: {IMPORT_ERROR!r}")
class BlimpPrefixControlTests(unittest.TestCase):
    def test_prepare_blimp_rows_keeps_valid_prefix_rows(self):
        tokenizer = _FakeTokenizer()
        df = pd.DataFrame(
            [
                {
                    "sentence_good": "Amanda was respected by some waitresses.",
                    "sentence_bad": "Amanda was respected by some picture.",
                    "one_prefix_prefix": "Amanda was respected by some",
                    "one_prefix_word_good": "waitresses",
                    "one_prefix_word_bad": "picture",
                    "field": "syntax",
                    "linguistics_term": "s-selection",
                    "UID": "animate_subject_passive",
                    "simple_LM_method": True,
                    "one_prefix_method": True,
                    "pairID": "0",
                }
            ]
        )

        rows, failures = prepare_blimp_passive_prefix_rows(df, tokenizer)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(failures), 0)
        self.assertEqual(rows.loc[0, "prefix"], "Amanda was respected by some")
        self.assertEqual(rows.loc[0, "seq_len"], 5)

    def test_prepare_blimp_rows_filters_empty_prefix(self):
        tokenizer = _FakeTokenizer()
        df = pd.DataFrame(
            [
                {
                    "sentence_good": "Amanda was respected by some waitresses.",
                    "sentence_bad": "Amanda was respected by some picture.",
                    "one_prefix_prefix": "   ",
                    "one_prefix_word_good": "waitresses",
                    "one_prefix_word_bad": "picture",
                    "field": "syntax",
                    "linguistics_term": "s-selection",
                    "UID": "animate_subject_passive",
                    "simple_LM_method": True,
                    "one_prefix_method": True,
                    "pairID": "1",
                }
            ]
        )

        rows, failures = prepare_blimp_passive_prefix_rows(df, tokenizer)

        self.assertEqual(len(rows), 0)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures.loc[0, "failure_reason"], "empty_prefix")

    def test_summarize_blimp_prefix_rows(self):
        rows = pd.DataFrame(
            [
                {
                    "full_prefers_animate": True,
                    "circuit_prefers_animate": True,
                    "full_logit_diff": 2.0,
                    "circuit_logit_diff": 1.5,
                    "logit_diff_delta_circuit_minus_full": -0.5,
                    "flip_to_animate": False,
                    "flip_away_from_animate": False,
                },
                {
                    "full_prefers_animate": False,
                    "circuit_prefers_animate": True,
                    "full_logit_diff": -1.0,
                    "circuit_logit_diff": 0.25,
                    "logit_diff_delta_circuit_minus_full": 1.25,
                    "flip_to_animate": True,
                    "flip_away_from_animate": False,
                },
            ]
        )

        summary = summarize_blimp_passive_prefix_control(rows)

        self.assertEqual(summary["example_count"], 2)
        self.assertAlmostEqual(summary["full_model_accuracy"], 0.5)
        self.assertAlmostEqual(summary["circuit_accuracy"], 1.0)
        self.assertAlmostEqual(summary["accuracy_delta_circuit_minus_full"], 0.5)
        self.assertEqual(summary["circuit_fix_count"], 1)
        self.assertEqual(summary["circuit_break_count"], 0)


if __name__ == "__main__":
    unittest.main()

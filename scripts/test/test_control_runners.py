from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


EXECUTABLE_DIR = Path(__file__).resolve().parents[1] / "executable"
if str(EXECUTABLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTABLE_DIR))

try:
    import torch

    from control_runners import (
        add_source_pair_keys,
        build_preposition_control_dataframe,
        make_control_eap_normalized_recovery_vector_metric,
        prepare_verb_noise_control_rows,
        select_verb_noise_sigma,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    torch = None
    add_source_pair_keys = None
    build_preposition_control_dataframe = None
    make_control_eap_normalized_recovery_vector_metric = None
    prepare_verb_noise_control_rows = None
    select_verb_noise_sigma = None
    IMPORT_ERROR = exc


class FakeTokenizer:
    def __init__(self, multi_token_words: set[str] | None = None):
        self.multi_token_words = set(multi_token_words or set())

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        del add_special_tokens
        token_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        token_id = 1
        idx = 0
        length = len(text)
        while idx < length:
            if text[idx].isspace():
                idx += 1
                continue
            start = idx
            while idx < length and not text[idx].isspace():
                idx += 1
            end = idx
            word = text[start:end]
            if word in self.multi_token_words and len(word) > 1:
                split = start + max(1, len(word) // 2)
                token_ids.extend([token_id, token_id + 1])
                offsets.extend([(start, split), (split, end)])
                token_id += 2
            else:
                token_ids.append(token_id)
                offsets.append((start, end))
                token_id += 1
        payload = {"input_ids": token_ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return SimpleNamespace(**payload)


@unittest.skipIf(IMPORT_ERROR is not None, f"control runner imports unavailable: {IMPORT_ERROR!r}")
class ControlRunnerTests(unittest.TestCase):
    def test_build_preposition_control_dataframe_preserves_pairing_and_keys(self):
        tokenizer = FakeTokenizer()
        source = pd.DataFrame(
            [
                {
                    "uid": "u1",
                    "patient": "ball",
                    "clean_verb": "thrown",
                    "corrupt_verb": "dropped",
                    "clean_prefix": "The ball was thrown by the",
                    "corrupt_prefix": "The ball was dropped by the",
                },
                {
                    "uid": "u2",
                    "patient": "book",
                    "clean_verb": "carried",
                    "corrupt_verb": "moved",
                    "clean_prefix": "The book was carried by the",
                    "corrupt_prefix": "The book was moved by the",
                },
            ]
        )

        control = build_preposition_control_dataframe(source, tokenizer)

        self.assertEqual(len(control), len(source))
        self.assertEqual(control["uid"].tolist(), source["uid"].tolist())
        self.assertEqual(control["control_type"].tolist(), ["by_to_near", "by_to_near"])
        self.assertTrue(all(text.endswith(" near the") for text in control["clean_prefix"]))
        self.assertTrue(all(text.endswith(" near the") for text in control["corrupt_prefix"]))
        self.assertEqual(control["source_pair_key"].nunique(), len(source))

    def test_build_preposition_control_dataframe_rejects_multitoken_near(self):
        tokenizer = FakeTokenizer(multi_token_words={"near"})
        source = pd.DataFrame(
            [
                {
                    "uid": "u1",
                    "patient": "ball",
                    "clean_verb": "thrown",
                    "corrupt_verb": "dropped",
                    "clean_prefix": "The ball was thrown by the",
                    "corrupt_prefix": "The ball was dropped by the",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "not a single token"):
            build_preposition_control_dataframe(source, tokenizer)

    def test_build_preposition_control_dataframe_normalizes_duplicated_metadata_columns(self):
        tokenizer = FakeTokenizer()
        source = pd.DataFrame(
            [
                {
                    "uid_x": "u1",
                    "uid_y": None,
                    "patient_x": "ball",
                    "patient_y": None,
                    "clean_verb_x": "thrown",
                    "clean_verb_y": None,
                    "corrupt_verb_x": "dropped",
                    "corrupt_verb_y": None,
                    "clean_prefix": "The ball was thrown by the",
                    "corrupt_prefix": "The ball was dropped by the",
                }
            ]
        )

        control = build_preposition_control_dataframe(source, tokenizer)

        self.assertEqual(control.loc[0, "uid"], "u1")
        self.assertEqual(control.loc[0, "patient"], "ball")
        self.assertEqual(control.loc[0, "clean_verb"], "thrown")
        self.assertEqual(control.loc[0, "corrupt_verb"], "dropped")

    def test_add_source_pair_keys_preserves_existing_key(self):
        source = pd.DataFrame(
            [
                {
                    "uid": "u1",
                    "clean_prefix": "The ball was thrown by the",
                    "corrupt_prefix": "The ball was dropped by the",
                    "source_pair_key": "stable-key",
                }
            ]
        )

        keyed = add_source_pair_keys(source)

        self.assertEqual(keyed.loc[0, "source_pair_key"], "stable-key")

    def test_control_recovery_metric_allows_negative_margin(self):
        metric = make_control_eap_normalized_recovery_vector_metric(
            torch.tensor([0]),
            torch.tensor([1]),
        )
        logits = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
        clean_logits = torch.tensor([[[-1.0, 0.0]]], dtype=torch.float32)
        label = torch.tensor([1.0], dtype=torch.float32)

        values = metric(
            logits,
            clean_logits,
            torch.tensor([1]),
            label,
        )

        self.assertTrue(torch.isfinite(values).all())
        self.assertAlmostEqual(float(values.item()), 0.5, places=6)

    def test_select_verb_noise_sigma_uses_absolute_mean_margin(self):
        sweep = pd.DataFrame(
            [
                {
                    "sigma_multiplier": 0.5,
                    "sigma": 0.5,
                    "margin_mean": 0.30,
                    "absolute_mean_margin": 0.30,
                    "mean_absolute_margin": 0.30,
                },
                {
                    "sigma_multiplier": 1.0,
                    "sigma": 1.0,
                    "margin_mean": -0.01,
                    "absolute_mean_margin": 0.01,
                    "mean_absolute_margin": 0.40,
                },
            ]
        )

        selected = select_verb_noise_sigma(sweep, tolerance=0.0)

        self.assertEqual(float(selected["sigma_multiplier"]), 1.0)
        self.assertAlmostEqual(float(selected["absolute_mean_margin"]), 0.01)

    def test_prepare_verb_noise_control_rows_rejects_multitoken_verbs(self):
        tokenizer = FakeTokenizer(multi_token_words={"dropped"})
        source = pd.DataFrame(
            [
                {
                    "uid": "u1",
                    "patient": "ball",
                    "clean_verb": "thrown",
                    "corrupt_verb": "dropped",
                    "clean_prefix": "The ball was thrown by the",
                    "corrupt_prefix": "The ball was dropped by the",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "single-token clean/corrupt verbs"):
            prepare_verb_noise_control_rows(source, tokenizer)


if __name__ == "__main__":
    unittest.main()

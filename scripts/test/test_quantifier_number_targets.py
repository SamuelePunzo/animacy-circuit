from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


DATASET_GENERATION_DIR = Path(__file__).resolve().parents[1] / "dataset_generation"
EXECUTABLE_DIR = Path(__file__).resolve().parents[1] / "executable"
for path in (DATASET_GENERATION_DIR, EXECUTABLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from create_quantifier_number_targets import build_quantifier_targets

    TARGET_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    build_quantifier_targets = None
    TARGET_IMPORT_ERROR = exc

try:
    from run_quantifier_number_discovery import tokenizer_filter_target_pairs

    RUNNER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    tokenizer_filter_target_pairs = None
    RUNNER_IMPORT_ERROR = exc


class FakeTokenizer:
    def __init__(self, multi_token_words: set[str] | None = None):
        self.multi_token_words = set(multi_token_words or set())
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        word = text.strip()
        if word in self.multi_token_words:
            return SimpleNamespace(input_ids=[101, 102])
        token_id = self.vocab.setdefault(word, len(self.vocab) + 1)
        return SimpleNamespace(input_ids=[token_id])

    def decode(self, token_ids):
        inverse = {value: key for key, value in self.vocab.items()}
        return inverse.get(int(token_ids[0]), f"<{int(token_ids[0])}>")


@unittest.skipIf(TARGET_IMPORT_ERROR is not None, f"target builder import unavailable: {TARGET_IMPORT_ERROR!r}")
class QuantifierTargetBuilderTests(unittest.TestCase):
    def test_allows_irregular_plural_when_roundtrip_is_valid(self):
        payload = build_quantifier_targets(
            {
                "animate": ["person", "officer"],
                "inanimate": ["city", "box"],
            }
        )

        animate_pairs = {(row["singular"], row["plural"]) for row in payload["targets"]["animate"]}
        inanimate_pairs = {(row["singular"], row["plural"]) for row in payload["targets"]["inanimate"]}

        self.assertIn(("person", "people"), animate_pairs)
        self.assertIn(("city", "cities"), inanimate_pairs)

    def test_rejects_singular_equal_plural(self):
        payload = build_quantifier_targets(
            {
                "animate": ["sheep", "officer"],
                "inanimate": ["series", "box"],
            }
        )

        rejected = {
            (target_class, row["word"], row["reason"])
            for target_class, rows in payload["rejections"].items()
            for row in rows
        }
        self.assertIn(("animate", "sheep", "singular_equals_plural"), rejected)
        self.assertIn(("inanimate", "series", "singular_equals_plural"), rejected)

    def test_rejects_plural_singular_collisions(self):
        payload = build_quantifier_targets(
            {
                "animate": ["person"],
                "inanimate": ["people"],
            }
        )

        self.assertEqual(payload["targets"]["animate"], [])
        self.assertEqual(payload["targets"]["inanimate"], [])
        reasons = [
            reason
            for rows in payload["rejections"].values()
            for row in rows
            for reason in row.get("collision_reasons", [])
        ]
        self.assertIn("plural_collides_with_singular", reasons)


@unittest.skipIf(RUNNER_IMPORT_ERROR is not None, f"runner import unavailable: {RUNNER_IMPORT_ERROR!r}")
class QuantifierTokenizerFilterTests(unittest.TestCase):
    def test_keeps_pair_only_when_singular_and_plural_are_one_token(self):
        kept, dropped = tokenizer_filter_target_pairs(
            [
                {"singular": "officer", "plural": "officers"},
                {"singular": "judge", "plural": "judges"},
            ],
            FakeTokenizer(multi_token_words={"judges"}),
        )

        self.assertEqual([row["singular"] for row in kept], ["officer"])
        self.assertEqual(dropped[0]["reason"], "plural_not_one_token")

    def test_drops_duplicate_token_ids(self):
        tokenizer = FakeTokenizer()
        tokenizer.vocab["officer"] = 1
        tokenizer.vocab["guard"] = 1

        kept, dropped = tokenizer_filter_target_pairs(
            [
                {"singular": "officer", "plural": "officers"},
                {"singular": "guard", "plural": "guards"},
            ],
            tokenizer,
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped[0]["reason"], "duplicate_singular_token_id")


if __name__ == "__main__":
    unittest.main()

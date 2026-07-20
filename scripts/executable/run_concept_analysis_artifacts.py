from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from circuit_finder_core import (
    add_concept_verb_positions,
    add_sequence_lengths,
    concept_steering_vector,
    generate_exact_length_batches,
    grouped_model_results_root,
    load_model,
    make_concept_steering_hook,
    normalize_concept_pair_metadata,
    random_control_vector,
    resolve_animacy_circuit_root,
    token_span_from_offsets,
    tokenizer_offsets,
)
from utils import save_csv, save_json, timestamp_tag

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")
DEFAULT_BLIMP_CONFIG = "animate_subject_trans"
PASSIVE_TRAILING_PARTICLES = {
    "about",
    "to",
    "for",
    "with",
    "on",
    "in",
    "into",
    "onto",
    "upon",
    "over",
    "under",
    "off",
    "up",
    "down",
    "around",
    "through",
    "across",
    "after",
    "before",
    "from",
}
AUXILIARY_OR_MODAL_TOKENS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
}
BLIMP_ALIGNMENT_FAILURE_COLUMNS = [
    "row",
    "UID",
    "sentence_good",
    "sentence_bad",
    "alignment_error",
    "good_subject_error",
    "bad_subject_error",
    "good_verb_error",
    "bad_verb_error",
]
BLIMP_RESULT_COLUMNS = [
    "UID",
    "sentence_good",
    "sentence_bad",
    "subject_head_good",
    "subject_head_bad",
    "main_verb",
    "condition",
    "control_label",
    "repeat",
    "good_logprob_before",
    "bad_logprob_before",
    "good_logprob_after",
    "bad_logprob_after",
    "baseline_score",
    "steered_score",
    "score_shift",
    "baseline_prefers_good",
    "steered_prefers_good",
    "preference_flipped_to_good",
    "preference_flipped_away_from_good",
    "hook_name",
    "alpha",
    "vector_norm",
]
BLIMP_SUMMARY_COLUMNS = [
    "condition",
    "control_label",
    "repeat",
    "example_count",
    "baseline_score_mean",
    "steered_score_mean",
    "score_shift_mean",
    "score_shift_std",
    "baseline_accuracy",
    "steered_accuracy",
    "accuracy_shift",
    "baseline_correct_count",
    "baseline_wrong_count",
    "flip_to_good_count",
    "flip_away_from_good_count",
    "flip_to_good_rate",
    "flip_away_from_good_rate",
    "flip_to_good_given_baseline_wrong",
    "flip_away_given_baseline_correct",
]


def is_passive_blimp_config(blimp_config: str) -> bool:
    return blimp_config == "animate_subject_passive"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute heavyweight concept-extraction analysis artifacts so the "
            "notebook only needs to load saved results."
        )
    )
    parser.add_argument("--model", default="gpt2", help="Model slug under results/concept_extraction.")
    parser.add_argument(
        "--day",
        default=None,
        help=(
            "Concept-extraction run directory to analyze, for example 2026-05-30 or smoke_test. "
            "When omitted, the latest run directory for the model is used."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional explicit concept-extraction run directory. Overrides --model/--day directory selection.",
    )
    parser.add_argument("--start-path", default=None)
    parser.add_argument("--activation-batch-size", type=int, default=64)
    parser.add_argument("--blimp-batch-size", type=int, default=32)
    parser.add_argument("--blimp-config", default=DEFAULT_BLIMP_CONFIG)
    parser.add_argument(
        "--blimp-random-control-repeats",
        type=int,
        default=None,
        help="Override the saved run's random-control repeat count for BLiMP controls.",
    )
    parser.add_argument(
        "--blimp-validation-size",
        type=int,
        default=100,
        help=(
            "Number of BLiMP rows to reserve for alpha selection. "
            "Use 0 to disable BLiMP-side alpha tuning and reuse the source alpha directly."
        ),
    )
    parser.add_argument(
        "--blimp-alpha-grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional alpha grid for BLiMP validation selection. "
            "Defaults to the source concept-extraction alpha grid."
        ),
    )
    parser.add_argument(
        "--blimp-selection-effect-fraction",
        type=float,
        default=0.9,
        help=(
            "Conservative alpha-selection threshold on the BLiMP validation sweep: "
            "pick the smallest |alpha| within this fraction of the best validation accuracy shift."
        ),
    )
    parser.add_argument(
        "--skip-activation-geometry",
        action="store_true",
        help="Skip activation/projection artifacts.",
    )
    parser.add_argument(
        "--skip-blimp",
        action="store_true",
        help="Skip BLiMP transfer artifacts.",
    )
    return parser


def latest_child_dir(path: Path) -> Path:
    dirs = [child for child in path.iterdir() if child.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under {path}")
    return max(dirs, key=lambda item: item.stat().st_mtime)


def latest_file(run_dir: Path, pattern: str, *, required: bool = False) -> Path | None:
    matches = sorted(run_dir.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        if required:
            raise FileNotFoundError(f"No file matching {pattern!r} in {run_dir}")
        return None
    return matches[-1]


def read_json_required(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(
    *,
    project_root: Path,
    model_slug: str,
    day: str | None,
    run_dir: str | None,
) -> Path:
    if run_dir is not None:
        resolved = Path(run_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {resolved}")
        return resolved
    model_root = grouped_model_results_root(project_root, "concept_extraction", model_slug)
    return model_root / day if day is not None else latest_child_dir(model_root)


def tokenize_with_spans(text: str) -> list[dict[str, object]]:
    return [
        {"text": match.group(0), "start": int(match.start()), "end": int(match.end())}
        for match in TOKEN_PATTERN.finditer(text)
    ]


def trailing_shared_token_count(good_tokens: Sequence[str], bad_tokens: Sequence[str]) -> int:
    shared = 0
    for good_token, bad_token in zip(reversed(good_tokens), reversed(bad_tokens)):
        if good_token != bad_token:
            break
        shared += 1
    return shared


def is_word_token(token_text: str) -> bool:
    return any(char.isalnum() for char in token_text)


def resolve_single_token_span(
    tokenizer,
    text: str,
    token: dict[str, object],
) -> tuple[tuple[int, int] | None, str | None]:
    offsets = tokenizer_offsets(tokenizer, text)
    if offsets is None:
        return None, "offset_mapping_unavailable"
    span = token_span_from_offsets(offsets, int(token["start"]), int(token["end"]))
    if span is None:
        return None, "token_span_missing"
    if (span[1] - span[0]) != 1:
        return span, "token_not_single_model_token"
    return span, None


def trans_main_verb_token(tokens: list[dict[str, object]], start_idx: int) -> dict[str, object] | None:
    idx = start_idx
    while idx < len(tokens):
        candidate = tokens[idx]
        text = str(candidate["text"]).lower()
        if not is_word_token(text):
            idx += 1
            continue
        if text in AUXILIARY_OR_MODAL_TOKENS or text in {"not", "n't"}:
            idx += 1
            continue
        return candidate
    idx = start_idx
    while idx < len(tokens):
        candidate = tokens[idx]
        if is_word_token(str(candidate["text"])):
            return candidate
        idx += 1
    return None


def blimp_trans_alignment_details(row: pd.Series, tokenizer) -> dict[str, object]:
    good_text = str(row["sentence_good"])
    bad_text = str(row["sentence_bad"])
    good_tokens = tokenize_with_spans(good_text)
    bad_tokens = tokenize_with_spans(bad_text)
    good_token_text = [str(token["text"]) for token in good_tokens]
    bad_token_text = [str(token["text"]) for token in bad_tokens]
    shared_tail = trailing_shared_token_count(good_token_text, bad_token_text)

    details: dict[str, object] = {
        "sentence_good": good_text,
        "sentence_bad": bad_text,
        "alignment_ok": True,
        "alignment_error": None,
        "subject_head_text_good": None,
        "subject_head_text_bad": None,
        "main_verb_text": None,
        "good_subject_span": None,
        "bad_subject_span": None,
        "good_verb_span": None,
        "bad_verb_span": None,
        "good_subject_error": None,
        "bad_subject_error": None,
        "good_verb_error": None,
        "bad_verb_error": None,
    }

    if shared_tail < 2:
        details["alignment_ok"] = False
        details["alignment_error"] = "shared_tail_too_short"
        return details

    good_verb_idx = len(good_tokens) - shared_tail
    bad_verb_idx = len(bad_tokens) - shared_tail
    if good_verb_idx <= 0 or bad_verb_idx <= 0:
        details["alignment_ok"] = False
        details["alignment_error"] = "missing_subject_head"
        return details

    good_subject_idx = good_verb_idx - 1
    bad_subject_idx = bad_verb_idx - 1
    good_subject = good_tokens[good_subject_idx]
    bad_subject = bad_tokens[bad_subject_idx]
    good_verb = trans_main_verb_token(good_tokens, good_verb_idx)
    bad_verb = trans_main_verb_token(bad_tokens, bad_verb_idx)
    if good_verb is None or bad_verb is None:
        details["alignment_ok"] = False
        details["alignment_error"] = "missing_main_verb"
        return details

    if not is_word_token(str(good_subject["text"])) or not is_word_token(str(bad_subject["text"])):
        details["alignment_ok"] = False
        details["alignment_error"] = "subject_head_not_word"
        return details
    if str(good_verb["text"]) != str(bad_verb["text"]):
        details["alignment_ok"] = False
        details["alignment_error"] = "verb_mismatch"
        return details
    if not is_word_token(str(good_verb["text"])):
        details["alignment_ok"] = False
        details["alignment_error"] = "verb_not_word"
        return details

    good_subject_span, good_subject_error = resolve_single_token_span(tokenizer, good_text, good_subject)
    bad_subject_span, bad_subject_error = resolve_single_token_span(tokenizer, bad_text, bad_subject)
    good_verb_span, good_verb_error = resolve_single_token_span(tokenizer, good_text, good_verb)
    bad_verb_span, bad_verb_error = resolve_single_token_span(tokenizer, bad_text, bad_verb)

    details.update(
        {
            "subject_head_text_good": str(good_subject["text"]),
            "subject_head_text_bad": str(bad_subject["text"]),
            "main_verb_text": str(good_verb["text"]),
            "good_subject_span": good_subject_span,
            "bad_subject_span": bad_subject_span,
            "good_verb_span": good_verb_span,
            "bad_verb_span": bad_verb_span,
            "good_subject_error": good_subject_error,
            "bad_subject_error": bad_subject_error,
            "good_verb_error": good_verb_error,
            "bad_verb_error": bad_verb_error,
        }
    )

    if any(
        error is not None
        for error in (good_subject_error, bad_subject_error, good_verb_error, bad_verb_error)
    ):
        details["alignment_ok"] = False
        details["alignment_error"] = "model_token_alignment_failed"
        return details

    return details


def passive_subject_head_token(tokens: list[dict[str, object]]) -> dict[str, object] | None:
    for token in reversed(tokens):
        if is_word_token(str(token["text"])):
            return token
    return None


def passive_main_verb_token(tokens: list[dict[str, object]], by_index: int) -> dict[str, object] | None:
    idx = by_index - 1
    while idx >= 0 and str(tokens[idx]["text"]).lower() in PASSIVE_TRAILING_PARTICLES:
        idx -= 1
    while idx >= 0:
        candidate = tokens[idx]
        if is_word_token(str(candidate["text"])):
            return candidate
        idx -= 1
    return None


def blimp_passive_alignment_details(row: pd.Series, tokenizer) -> dict[str, object]:
    good_text = str(row["sentence_good"])
    bad_text = str(row["sentence_bad"])
    good_tokens = tokenize_with_spans(good_text)
    bad_tokens = tokenize_with_spans(bad_text)
    good_token_text = [str(token["text"]) for token in good_tokens]
    bad_token_text = [str(token["text"]) for token in bad_tokens]

    details: dict[str, object] = {
        "sentence_good": good_text,
        "sentence_bad": bad_text,
        "alignment_ok": True,
        "alignment_error": None,
        "subject_head_text_good": None,
        "subject_head_text_bad": None,
        "main_verb_text": None,
        "good_subject_span": None,
        "bad_subject_span": None,
        "good_verb_span": None,
        "bad_verb_span": None,
        "good_subject_error": None,
        "bad_subject_error": None,
        "good_verb_error": None,
        "bad_verb_error": None,
    }

    try:
        good_by_index = next(i for i, token in enumerate(good_token_text) if token.lower() == "by")
        bad_by_index = next(i for i, token in enumerate(bad_token_text) if token.lower() == "by")
    except StopIteration:
        details["alignment_ok"] = False
        details["alignment_error"] = "missing_by_phrase"
        return details

    if good_by_index != bad_by_index:
        details["alignment_ok"] = False
        details["alignment_error"] = "by_position_mismatch"
        return details

    good_subject = passive_subject_head_token(good_tokens[good_by_index + 1 :])
    bad_subject = passive_subject_head_token(bad_tokens[bad_by_index + 1 :])
    if good_subject is None or bad_subject is None:
        details["alignment_ok"] = False
        details["alignment_error"] = "missing_subject_head"
        return details

    good_verb = passive_main_verb_token(good_tokens, good_by_index)
    bad_verb = passive_main_verb_token(bad_tokens, bad_by_index)
    if good_verb is None or bad_verb is None:
        details["alignment_ok"] = False
        details["alignment_error"] = "missing_main_verb"
        return details

    if str(good_verb["text"]).lower() != str(bad_verb["text"]).lower():
        details["alignment_ok"] = False
        details["alignment_error"] = "verb_mismatch"
        return details

    good_subject_span, good_subject_error = resolve_single_token_span(tokenizer, good_text, good_subject)
    bad_subject_span, bad_subject_error = resolve_single_token_span(tokenizer, bad_text, bad_subject)
    good_verb_span, good_verb_error = resolve_single_token_span(tokenizer, good_text, good_verb)
    bad_verb_span, bad_verb_error = resolve_single_token_span(tokenizer, bad_text, bad_verb)

    details.update(
        {
            "subject_head_text_good": str(good_subject["text"]),
            "subject_head_text_bad": str(bad_subject["text"]),
            "main_verb_text": str(good_verb["text"]),
            "good_subject_span": good_subject_span,
            "bad_subject_span": bad_subject_span,
            "good_verb_span": good_verb_span,
            "bad_verb_span": bad_verb_span,
            "good_subject_error": good_subject_error,
            "bad_subject_error": bad_subject_error,
            "good_verb_error": good_verb_error,
            "bad_verb_error": bad_verb_error,
        }
    )

    if any(
        error is not None
        for error in (good_subject_error, bad_subject_error, good_verb_error, bad_verb_error)
    ):
        details["alignment_ok"] = False
        details["alignment_error"] = "model_token_alignment_failed"
        return details

    return details


def blimp_pair_alignment_details(row: pd.Series, tokenizer, blimp_config: str) -> dict[str, object]:
    if is_passive_blimp_config(blimp_config):
        return blimp_passive_alignment_details(row, tokenizer)
    return blimp_trans_alignment_details(row, tokenizer)


def add_blimp_subject_and_verb_positions(
    df: pd.DataFrame,
    tokenizer,
    *,
    blimp_config: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for row_idx, row in df.reset_index(drop=True).iterrows():
        details = blimp_pair_alignment_details(row, tokenizer, blimp_config)
        if is_passive_blimp_config(blimp_config):
            good_verb_span = details.get("good_verb_span")
            bad_verb_span = details.get("bad_verb_span")
            good_verb_error = details.get("good_verb_error")
            bad_verb_error = details.get("bad_verb_error")
            verb_alignment_ok = (
                details.get("main_verb_text") is not None
                and good_verb_span is not None
                and bad_verb_span is not None
                and good_verb_error is None
                and bad_verb_error is None
            )
            if not verb_alignment_ok:
                failures.append(
                    {
                        "row": int(row_idx),
                        "UID": row.get("UID"),
                        "sentence_good": row["sentence_good"],
                        "sentence_bad": row["sentence_bad"],
                        "alignment_error": details.get("alignment_error"),
                        "good_subject_error": details.get("good_subject_error"),
                        "bad_subject_error": details.get("bad_subject_error"),
                        "good_verb_error": good_verb_error,
                        "bad_verb_error": bad_verb_error,
                    }
                )
                continue

            subject_alignment_ok = (
                details.get("good_subject_span") is not None
                and details.get("bad_subject_span") is not None
                and details.get("good_subject_error") is None
                and details.get("bad_subject_error") is None
            )

            item = row.to_dict()
            item["subject_head_good"] = details["subject_head_text_good"]
            item["subject_head_bad"] = details["subject_head_text_bad"]
            item["main_verb"] = details["main_verb_text"]
            item["subject_alignment_ok"] = bool(subject_alignment_ok)
            item["good_subject_token_position"] = (
                int(details["good_subject_span"][0] + 1) if subject_alignment_ok else None
            )
            item["bad_subject_token_position"] = (
                int(details["bad_subject_span"][0] + 1) if subject_alignment_ok else None
            )
            item["good_verb_token_position"] = int(good_verb_span[0] + 1)
            item["bad_verb_token_position"] = int(bad_verb_span[0] + 1)
            rows.append(item)
            continue

        if not bool(details["alignment_ok"]):
            failures.append(
                {
                    "row": int(row_idx),
                    "UID": row.get("UID"),
                    "sentence_good": row["sentence_good"],
                    "sentence_bad": row["sentence_bad"],
                    "alignment_error": details.get("alignment_error"),
                    "good_subject_error": details.get("good_subject_error"),
                    "bad_subject_error": details.get("bad_subject_error"),
                    "good_verb_error": details.get("good_verb_error"),
                    "bad_verb_error": details.get("bad_verb_error"),
                }
            )
            continue

        item = row.to_dict()
        item["subject_head_good"] = details["subject_head_text_good"]
        item["subject_head_bad"] = details["subject_head_text_bad"]
        item["main_verb"] = details["main_verb_text"]
        item["good_subject_token_position"] = int(details["good_subject_span"][0] + 1)
        item["bad_subject_token_position"] = int(details["bad_subject_span"][0] + 1)
        item["good_verb_token_position"] = int(details["good_verb_span"][0] + 1)
        item["bad_verb_token_position"] = int(details["bad_verb_span"][0] + 1)
        rows.append(item)

    return pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(
        failures,
        columns=BLIMP_ALIGNMENT_FAILURE_COLUMNS,
    )


def sentence_length_no_special(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded.input_ids)


def score_sentences_with_optional_steering(
    model,
    texts: Sequence[str],
    positions: Sequence[int] | None = None,
    *,
    hook_name: str | None = None,
    vector: torch.Tensor | None = None,
    alpha: float | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros(0, dtype=np.float32)

    position_list = None if positions is None else [int(position) for position in positions]
    lengths = np.array([sentence_length_no_special(model.tokenizer, text) for text in texts], dtype=np.int64)
    scores = np.zeros(len(texts), dtype=np.float32)

    for seq_len in sorted(set(lengths.tolist())):
        group_indices = np.where(lengths == seq_len)[0]
        for start in range(0, len(group_indices), batch_size):
            batch_indices = group_indices[start : start + batch_size]
            batch_texts = [texts[int(idx)] for idx in batch_indices]
            batch_tokens = model.to_tokens(batch_texts, prepend_bos=True).to(model.cfg.device)
            with torch.no_grad():
                logits = model(batch_tokens)
                if hook_name is not None and vector is not None and alpha is not None:
                    batch_positions = [position_list[int(idx)] for idx in batch_indices]
                    logits = model.run_with_hooks(
                        batch_tokens,
                        fwd_hooks=[(hook_name, make_concept_steering_hook(batch_positions, vector, float(alpha)))],
                    )
            target_tokens = batch_tokens[:, 1:]
            token_logprobs = logits[:, :-1, :].log_softmax(dim=-1)
            sentence_logprobs = token_logprobs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            scores[batch_indices] = sentence_logprobs.detach().float().cpu().numpy()

    return scores


def evaluate_blimp_position_condition(
    model,
    df: pd.DataFrame,
    *,
    hook_name: str,
    vector: torch.Tensor,
    alpha: float,
    good_position_column: str,
    bad_position_column: str,
    condition_name: str,
    control_label: str,
    repeat: int | None = None,
    batch_size: int = 32,
    good_before: np.ndarray | None = None,
    bad_before: np.ndarray | None = None,
    steer_good: bool = True,
    steer_bad: bool = True,
) -> pd.DataFrame:
    good_sentences = df["sentence_good"].astype(str).tolist()
    bad_sentences = df["sentence_bad"].astype(str).tolist()
    good_positions = df[good_position_column].astype(int).tolist()
    bad_positions = df[bad_position_column].astype(int).tolist()

    if good_before is None:
        good_before = score_sentences_with_optional_steering(model, good_sentences, batch_size=batch_size)
    if bad_before is None:
        bad_before = score_sentences_with_optional_steering(model, bad_sentences, batch_size=batch_size)
    if steer_good:
        good_after = score_sentences_with_optional_steering(
            model,
            good_sentences,
            positions=good_positions,
            hook_name=hook_name,
            vector=vector,
            alpha=alpha,
            batch_size=batch_size,
        )
    else:
        good_after = np.array(good_before, copy=True)
    if steer_bad:
        bad_after = score_sentences_with_optional_steering(
            model,
            bad_sentences,
            positions=bad_positions,
            hook_name=hook_name,
            vector=vector,
            alpha=alpha,
            batch_size=batch_size,
        )
    else:
        bad_after = np.array(bad_before, copy=True)

    result = df[
        [
            "UID",
            "sentence_good",
            "sentence_bad",
            "subject_head_good",
            "subject_head_bad",
            "main_verb",
            good_position_column,
            bad_position_column,
        ]
    ].copy()
    result["condition"] = condition_name
    result["control_label"] = control_label
    result["repeat"] = repeat
    result["good_logprob_before"] = good_before
    result["bad_logprob_before"] = bad_before
    result["good_logprob_after"] = good_after
    result["bad_logprob_after"] = bad_after
    result["baseline_score"] = result["good_logprob_before"] - result["bad_logprob_before"]
    result["steered_score"] = result["good_logprob_after"] - result["bad_logprob_after"]
    result["score_shift"] = result["steered_score"] - result["baseline_score"]
    result["baseline_prefers_good"] = result["baseline_score"] > 0
    result["steered_prefers_good"] = result["steered_score"] > 0
    result["preference_flipped_to_good"] = (~result["baseline_prefers_good"]) & result["steered_prefers_good"]
    result["preference_flipped_away_from_good"] = result["baseline_prefers_good"] & (~result["steered_prefers_good"])
    result["hook_name"] = hook_name
    result["alpha"] = float(alpha)
    result["vector_norm"] = float(vector.norm().item())
    return result


def summarize_blimp_condition(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {
            "condition": None,
            "control_label": None,
            "repeat": None,
            "example_count": 0,
            "baseline_score_mean": 0.0,
            "steered_score_mean": 0.0,
            "score_shift_mean": 0.0,
            "score_shift_std": 0.0,
            "baseline_accuracy": 0.0,
            "steered_accuracy": 0.0,
            "accuracy_shift": 0.0,
            "baseline_correct_count": 0,
            "baseline_wrong_count": 0,
            "flip_to_good_count": 0,
            "flip_away_from_good_count": 0,
            "flip_to_good_rate": 0.0,
            "flip_away_from_good_rate": 0.0,
            "flip_to_good_given_baseline_wrong": 0.0,
            "flip_away_given_baseline_correct": 0.0,
        }
    baseline_correct = rows["baseline_prefers_good"]
    baseline_wrong = ~baseline_correct
    flip_to_good = rows["preference_flipped_to_good"]
    flip_away = rows["preference_flipped_away_from_good"]
    baseline_correct_count = int(baseline_correct.sum())
    baseline_wrong_count = int(baseline_wrong.sum())
    flip_to_good_count = int(flip_to_good.sum())
    flip_away_count = int(flip_away.sum())
    return {
        "condition": rows["condition"].iloc[0],
        "control_label": rows["control_label"].iloc[0],
        "repeat": rows["repeat"].iloc[0],
        "example_count": int(len(rows)),
        "baseline_score_mean": float(rows["baseline_score"].mean()),
        "steered_score_mean": float(rows["steered_score"].mean()),
        "score_shift_mean": float(rows["score_shift"].mean()),
        "score_shift_std": float(rows["score_shift"].std(ddof=0)) if len(rows) > 1 else 0.0,
        "baseline_accuracy": float(rows["baseline_prefers_good"].mean()),
        "steered_accuracy": float(rows["steered_prefers_good"].mean()),
        "accuracy_shift": float(rows["steered_prefers_good"].mean() - rows["baseline_prefers_good"].mean()),
        "baseline_correct_count": baseline_correct_count,
        "baseline_wrong_count": baseline_wrong_count,
        "flip_to_good_count": flip_to_good_count,
        "flip_away_from_good_count": flip_away_count,
        "flip_to_good_rate": float(flip_to_good.mean()),
        "flip_away_from_good_rate": float(flip_away.mean()),
        "flip_to_good_given_baseline_wrong": (
            float(flip_to_good_count / baseline_wrong_count) if baseline_wrong_count else 0.0
        ),
        "flip_away_given_baseline_correct": (
            float(flip_away_count / baseline_correct_count) if baseline_correct_count else 0.0
        ),
    }


def blimp_alpha_selection_condition(blimp_config: str) -> str:
    return "main_verb" if is_passive_blimp_config(blimp_config) else "subject_head"


def split_blimp_validation_rows(
    df: pd.DataFrame,
    *,
    validation_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if validation_size <= 0 or df.empty:
        return df.iloc[0:0].copy(), df.reset_index(drop=True).copy()

    max_validation = max(len(df) - 1, 0)
    used_validation_size = min(int(validation_size), max_validation)
    if used_validation_size <= 0:
        return df.iloc[0:0].copy(), df.reset_index(drop=True).copy()

    shuffled = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    validation_df = shuffled.iloc[:used_validation_size].reset_index(drop=True).copy()
    test_df = shuffled.iloc[used_validation_size:].reset_index(drop=True).copy()
    return validation_df, test_df


def select_conservative_blimp_alpha(
    validation_sweep: pd.DataFrame,
    *,
    effect_fraction: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    if validation_sweep.empty:
        raise ValueError("Cannot select a BLiMP alpha from an empty validation sweep.")
    if not 0 < effect_fraction <= 1:
        raise ValueError("blimp_selection_effect_fraction must be in (0, 1].")

    sweep = validation_sweep.copy()
    sweep["abs_alpha"] = sweep["alpha"].abs()
    best_effect = float(sweep["accuracy_shift"].max())
    threshold = best_effect * float(effect_fraction) if best_effect > 0 else best_effect
    sweep["selection_threshold"] = threshold
    sweep["selection_eligible"] = sweep["accuracy_shift"] >= threshold
    eligible = sweep.loc[sweep["selection_eligible"]].copy()
    if eligible.empty:
        eligible = sweep.copy()
    selected = eligible.sort_values(
        ["abs_alpha", "accuracy_shift", "steered_accuracy", "score_shift_mean"],
        ascending=[True, False, False, False],
    ).iloc[0].to_dict()
    sweep["selected"] = (
        (sweep["alpha"] == selected["alpha"])
        & (sweep["accuracy_shift"] == selected["accuracy_shift"])
        & (sweep["steered_accuracy"] == selected["steered_accuracy"])
    )
    return selected, sweep.sort_values(
        ["accuracy_shift", "steered_accuracy", "score_shift_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def find_optional_shuffled_vector_path(run_dir: Path) -> Path | None:
    patterns = (
        "shuffled_label_vectors_*.pt",
        "concept_vectors_shuffled_labels_*.pt",
        "*shuffled*label*vector*.pt",
        "*shuffle*vector*.pt",
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(run_dir.glob(pattern))
    if not matches:
        return None
    return sorted(set(matches), key=lambda path: path.stat().st_mtime)[-1]


def compute_activation_geometry(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    selected_hook: str,
    steering_vector: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    split_path = latest_file(run_dir, "test_split_*.csv", required=True)
    assert split_path is not None
    test_split = pd.read_csv(split_path)
    test_activation_df = normalize_concept_pair_metadata(test_split)

    model = load_model(summary.get("model_name", "gpt2"))
    test_activation_df = add_sequence_lengths(test_activation_df, model)
    test_activation_df = add_concept_verb_positions(test_activation_df, model.tokenizer)

    clean_acts: list[torch.Tensor] = []
    corrupt_acts: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []

    for clean_tokens, corrupt_tokens, batch_df in generate_exact_length_batches(
        test_activation_df,
        model,
        batch_size,
        model.cfg.device,
    ):
        positions = torch.tensor(
            batch_df["verb_token_position"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=model.cfg.device,
        )
        batch_indices = torch.arange(clean_tokens.shape[0], device=model.cfg.device)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=[selected_hook])
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=[selected_hook])
        clean_selected = clean_cache[selected_hook][batch_indices, positions, :].detach().float().cpu()
        corrupt_selected = corrupt_cache[selected_hook][batch_indices, positions, :].detach().float().cpu()
        clean_acts.append(clean_selected)
        corrupt_acts.append(corrupt_selected)
        records.extend(batch_df.to_dict("records"))
        del clean_cache, corrupt_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    clean_acts_tensor = torch.cat(clean_acts, dim=0)
    corrupt_acts_tensor = torch.cat(corrupt_acts, dim=0)
    pair_diffs = clean_acts_tensor - corrupt_acts_tensor

    projection_df = pd.DataFrame(records)
    projection_df["clean_proj"] = (clean_acts_tensor @ steering_vector).numpy()
    projection_df["corrupt_proj"] = (corrupt_acts_tensor @ steering_vector).numpy()
    projection_df["pair_diff_proj"] = (pair_diffs @ steering_vector).numpy()
    projection_df["clean_minus_corrupt_l2"] = pair_diffs.norm(dim=1).numpy()

    projection_summary = {
        "clean_proj_mean": float(projection_df["clean_proj"].mean()),
        "corrupt_proj_mean": float(projection_df["corrupt_proj"].mean()),
        "pair_diff_proj_mean": float(projection_df["pair_diff_proj"].mean()),
        "pair_diff_proj_std": float(projection_df["pair_diff_proj"].std(ddof=0)) if len(projection_df) > 1 else 0.0,
        "test_steering_signed_effect_mean": summary.get("test_summary", {}).get("signed_effect_mean"),
        "random_control_signed_effect_mean": summary.get("random_control_summary", {}).get("signed_effect_mean"),
        "example_count": int(len(projection_df)),
        "selected_hook": selected_hook,
    }

    pca_status = {"available": False, "error": None}
    pca_points = pd.DataFrame()
    if PCA is None:
        pca_status["error"] = "scikit-learn unavailable"
    else:
        try:
            combined = torch.cat([clean_acts_tensor, corrupt_acts_tensor], dim=0).numpy()
            coords = PCA(n_components=2).fit_transform(combined)
            n = len(projection_df)
            clean_points = projection_df.copy().reset_index().rename(columns={"index": "pair_id"})
            clean_points["pc1"] = coords[:n, 0]
            clean_points["pc2"] = coords[:n, 1]
            clean_points["sentence_type"] = "clean"
            clean_points["active_verb"] = clean_points["clean_verb"]
            clean_points["active_sentence"] = clean_points["clean_prefix"]

            corrupt_points = projection_df.copy().reset_index().rename(columns={"index": "pair_id"})
            corrupt_points["pc1"] = coords[n:, 0]
            corrupt_points["pc2"] = coords[n:, 1]
            corrupt_points["sentence_type"] = "corrupt"
            corrupt_points["active_verb"] = corrupt_points["corrupt_verb"]
            corrupt_points["active_sentence"] = corrupt_points["corrupt_prefix"]

            pca_points = pd.concat([clean_points, corrupt_points], ignore_index=True)
            pca_status["available"] = True
        except Exception as exc:
            pca_status["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "projection_rows": projection_df,
        "projection_summary": projection_summary,
        "pca_points": pca_points,
        "pca_status": pca_status,
        "device": str(model.cfg.device),
        "test_examples": int(len(test_activation_df)),
    }


def compute_blimp_transfer(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    selected_hook: str,
    steering_vector: torch.Tensor,
    source_selected_alpha: float,
    normalize_flag: bool,
    batch_size: int,
    blimp_config: str,
    random_control_repeats: int,
    validation_size: int,
    alpha_grid: Sequence[float],
    selection_effect_fraction: float,
) -> dict[str, Any]:
    from datasets import load_dataset

    model = load_model(summary.get("model_name", "gpt2"))
    raw_blimp = load_dataset("nyu-mll/blimp", blimp_config, split="train")
    blimp_dataset = raw_blimp.to_pandas()
    if "simple_LM_method" in blimp_dataset.columns:
        blimp_dataset = blimp_dataset[blimp_dataset["simple_LM_method"].fillna(False)].reset_index(drop=True)

    aligned_rows, alignment_failures = add_blimp_subject_and_verb_positions(
        blimp_dataset,
        model.tokenizer,
        blimp_config=blimp_config,
    )
    seed = int((summary.get("config") or {}).get("seed") or 42)
    validation_rows, evaluation_rows = split_blimp_validation_rows(
        aligned_rows,
        validation_size=validation_size,
        seed=seed,
    )
    alpha_values = [float(alpha) for alpha in alpha_grid]
    if not alpha_values:
        alpha_values = [float(source_selected_alpha)]

    status = {
        "config": blimp_config,
        "ready": not aligned_rows.empty,
        "reason": None if not aligned_rows.empty else "No BLiMP rows survived token alignment.",
        "raw_rows": int(len(blimp_dataset)),
        "aligned_rows": int(len(aligned_rows)),
        "alignment_failures": int(len(alignment_failures)),
        "validation_rows": int(len(validation_rows)),
        "evaluation_rows": int(len(evaluation_rows)),
        "position_policy": (
            "main_verb_single_token_only" if is_passive_blimp_config(blimp_config) else "subject_and_verb_single_token"
        ),
        "selected_hook": selected_hook,
        "source_selected_alpha": float(source_selected_alpha),
        "selected_alpha": float(source_selected_alpha),
        "alpha_grid": alpha_values,
        "alpha_selection_condition": blimp_alpha_selection_condition(blimp_config),
        "alpha_selection_effect_fraction": float(selection_effect_fraction),
        "random_control_repeats": int(random_control_repeats),
        "device": str(model.cfg.device),
    }
    if aligned_rows.empty:
        return {
            "status": status,
            "alignment_failures": alignment_failures,
            "results": pd.DataFrame(columns=BLIMP_RESULT_COLUMNS),
            "summary": pd.DataFrame(columns=BLIMP_SUMMARY_COLUMNS),
            "random_summary": pd.DataFrame(columns=BLIMP_SUMMARY_COLUMNS),
            "validation_sweep": pd.DataFrame(),
        }

    passive_config = is_passive_blimp_config(blimp_config)
    validation_sweep = pd.DataFrame()
    selected_alpha = float(source_selected_alpha)
    if not validation_rows.empty:
        validation_good_before = score_sentences_with_optional_steering(
            model,
            validation_rows["sentence_good"].astype(str).tolist(),
            batch_size=batch_size,
        )
        validation_bad_before = score_sentences_with_optional_steering(
            model,
            validation_rows["sentence_bad"].astype(str).tolist(),
            batch_size=batch_size,
        )
        validation_rows_by_alpha: list[dict[str, object]] = []
        selection_condition = blimp_alpha_selection_condition(blimp_config)
        for alpha in alpha_values:
            if selection_condition == "main_verb":
                condition_rows = evaluate_blimp_position_condition(
                    model,
                    validation_rows,
                    hook_name=selected_hook,
                    vector=steering_vector,
                    alpha=float(alpha),
                    good_position_column="good_verb_token_position",
                    bad_position_column="bad_verb_token_position",
                    condition_name="main_verb",
                    control_label="validation_alpha_sweep",
                    batch_size=batch_size,
                    good_before=validation_good_before,
                    bad_before=validation_bad_before,
                )
            else:
                condition_rows = evaluate_blimp_position_condition(
                    model,
                    validation_rows,
                    hook_name=selected_hook,
                    vector=steering_vector,
                    alpha=float(alpha),
                    good_position_column="good_subject_token_position",
                    bad_position_column="bad_subject_token_position",
                    condition_name="subject_head",
                    control_label="validation_alpha_sweep",
                    batch_size=batch_size,
                    good_before=validation_good_before,
                    bad_before=validation_bad_before,
                )
            validation_rows_by_alpha.append(
                {
                    **summarize_blimp_condition(condition_rows),
                    "alpha": float(alpha),
                    "hook_name": selected_hook,
                    "vector_norm": float(steering_vector.norm().item()),
                }
            )
        selected_alpha_info, validation_sweep = select_conservative_blimp_alpha(
            pd.DataFrame(validation_rows_by_alpha),
            effect_fraction=selection_effect_fraction,
        )
        selected_alpha = float(selected_alpha_info["alpha"])
        status["selected_alpha"] = selected_alpha
        status["validation_selection"] = selected_alpha_info
    else:
        status["validation_selection"] = None

    if evaluation_rows.empty:
        status["ready"] = False
        status["reason"] = "No BLiMP rows remain for evaluation after reserving validation rows."
        return {
            "status": status,
            "alignment_failures": alignment_failures,
            "results": pd.DataFrame(columns=BLIMP_RESULT_COLUMNS),
            "summary": pd.DataFrame(columns=BLIMP_SUMMARY_COLUMNS),
            "random_summary": pd.DataFrame(columns=BLIMP_SUMMARY_COLUMNS),
            "validation_sweep": validation_sweep,
        }

    baseline_good = score_sentences_with_optional_steering(
        model,
        evaluation_rows["sentence_good"].astype(str).tolist(),
        batch_size=batch_size,
    )
    baseline_bad = score_sentences_with_optional_steering(
        model,
        evaluation_rows["sentence_bad"].astype(str).tolist(),
        batch_size=batch_size,
    )

    result_frames: list[pd.DataFrame] = []
    if not passive_config:
        result_frames.append(
            evaluate_blimp_position_condition(
                model,
                evaluation_rows,
                hook_name=selected_hook,
                vector=steering_vector,
                alpha=float(selected_alpha),
                good_position_column="good_subject_token_position",
                bad_position_column="bad_subject_token_position",
                condition_name="subject_head",
                control_label="selected_direction",
                batch_size=batch_size,
                good_before=baseline_good,
                bad_before=baseline_bad,
            )
        )
    result_frames.extend(
        [
            evaluate_blimp_position_condition(
                model,
                evaluation_rows,
                hook_name=selected_hook,
                vector=steering_vector,
                alpha=float(selected_alpha),
                good_position_column="good_verb_token_position",
                bad_position_column="bad_verb_token_position",
                condition_name="main_verb",
                control_label="selected_direction",
                batch_size=batch_size,
                good_before=baseline_good,
                bad_before=baseline_bad,
            ),
            evaluate_blimp_position_condition(
                model,
                evaluation_rows,
                hook_name=selected_hook,
                vector=steering_vector,
                alpha=float(selected_alpha),
                good_position_column="good_verb_token_position",
                bad_position_column="bad_verb_token_position",
                condition_name="main_verb_bad_only",
                control_label="selected_direction",
                batch_size=batch_size,
                good_before=baseline_good,
                bad_before=baseline_bad,
                steer_good=False,
                steer_bad=True,
            ),
        ]
    )

    random_rows: list[dict[str, object]] = []
    for repeat_index in range(int(random_control_repeats)):
        control_vector = random_control_vector(
            steering_vector,
            seed=seed,
            repeat_index=repeat_index,
        )
        random_result = evaluate_blimp_position_condition(
            model,
            evaluation_rows,
            hook_name=selected_hook,
            vector=control_vector,
            alpha=float(selected_alpha),
            good_position_column="good_verb_token_position" if passive_config else "good_subject_token_position",
            bad_position_column="bad_verb_token_position" if passive_config else "bad_subject_token_position",
            condition_name="main_verb" if passive_config else "subject_head",
            control_label="random_direction",
            repeat=repeat_index,
            batch_size=batch_size,
            good_before=baseline_good,
            bad_before=baseline_bad,
        )
        result_frames.append(random_result)
        random_rows.append(summarize_blimp_condition(random_result))

    shuffled_vector_path = find_optional_shuffled_vector_path(run_dir)
    status["shuffled_vector_path"] = str(shuffled_vector_path) if shuffled_vector_path is not None else None
    if shuffled_vector_path is not None:
        shuffled_payload = torch.load(shuffled_vector_path, map_location="cpu")
        shuffled_raw = shuffled_payload.get("vectors", {}).get(selected_hook)
        if shuffled_raw is not None:
            shuffled_vector = concept_steering_vector(shuffled_raw.float().cpu(), normalize=normalize_flag)
            result_frames.append(
                evaluate_blimp_position_condition(
                    model,
                    evaluation_rows,
                    hook_name=selected_hook,
                    vector=shuffled_vector,
                    alpha=float(selected_alpha),
                    good_position_column="good_verb_token_position" if passive_config else "good_subject_token_position",
                    bad_position_column="bad_verb_token_position" if passive_config else "bad_subject_token_position",
                    condition_name="main_verb" if passive_config else "subject_head",
                    control_label="shuffled_label_direction",
                    batch_size=batch_size,
                    good_before=baseline_good,
                    bad_before=baseline_bad,
                )
            )

    results = pd.concat(result_frames, ignore_index=True)
    summary_rows = []
    for (_, _, _), group in results.groupby(["condition", "control_label", "repeat"], dropna=False):
        summary_rows.append(summarize_blimp_condition(group.reset_index(drop=True)))

    return {
        "status": status,
        "alignment_failures": alignment_failures,
        "results": results,
        "summary": pd.DataFrame(summary_rows).sort_values(
            ["condition", "control_label", "repeat"],
            na_position="last",
        ).reset_index(drop=True),
        "random_summary": pd.DataFrame(random_rows, columns=BLIMP_SUMMARY_COLUMNS),
        "validation_sweep": validation_sweep,
    }


def main() -> None:
    args = make_parser().parse_args()
    project_root = resolve_animacy_circuit_root(args.start_path)
    run_dir = resolve_run_dir(
        project_root=project_root,
        model_slug=args.model,
        day=args.day,
        run_dir=args.run_dir,
    )

    summary_path = latest_file(run_dir, "concept_extraction_summary_*.json", required=True)
    selected_path = latest_file(run_dir, "selected_site_*.json", required=True)
    vector_path = latest_file(run_dir, "concept_vectors_*.pt", required=True)
    assert summary_path is not None and selected_path is not None and vector_path is not None

    summary = read_json_required(summary_path)
    selected_payload = read_json_required(selected_path)
    selected_info = summary.get("selected") or selected_payload.get("selected") or {}
    selected_hook = selected_info.get("hook_name")
    source_selected_alpha = float(selected_info.get("alpha"))
    if not selected_hook:
        raise ValueError(f"No selected hook found in {summary_path}")

    vector_payload = torch.load(vector_path, map_location="cpu")
    raw_vector = vector_payload["vectors"][selected_hook].float().cpu()
    normalize_flag = bool((summary.get("config") or {}).get("normalize_concept_vector", True))
    steering_vector = concept_steering_vector(raw_vector, normalize=normalize_flag)

    analysis_tag = timestamp_tag()
    manifest: dict[str, Any] = {
        "analysis_tag": analysis_tag,
        "run_dir": str(run_dir),
        "source_summary": str(summary_path),
        "source_selected_site": str(selected_path),
        "source_vectors": str(vector_path),
        "selected_hook": selected_hook,
        "source_selected_alpha": float(source_selected_alpha),
        "raw_vector_norm": float(raw_vector.norm().item()),
        "steering_vector_norm": float(steering_vector.norm().item()),
        "normalize_concept_vector": normalize_flag,
        "paths": {},
    }

    if not args.skip_activation_geometry:
        geometry = compute_activation_geometry(
            run_dir=run_dir,
            summary=summary,
            selected_hook=selected_hook,
            steering_vector=steering_vector,
            batch_size=args.activation_batch_size,
        )
        projection_rows_path = run_dir / f"analysis_projection_rows_{analysis_tag}.csv"
        projection_summary_path = run_dir / f"analysis_projection_summary_{analysis_tag}.json"
        pca_points_path = run_dir / f"analysis_pca_points_{analysis_tag}.csv"
        save_csv(geometry["projection_rows"], projection_rows_path, index=False)
        save_json(
            projection_summary_path,
            {
                **geometry["projection_summary"],
                "pca_status": geometry["pca_status"],
                "device": geometry["device"],
                "test_examples": geometry["test_examples"],
            },
        )
        if not geometry["pca_points"].empty:
            save_csv(geometry["pca_points"], pca_points_path, index=False)
            manifest["paths"]["analysis_pca_points"] = str(pca_points_path)
        manifest["paths"]["analysis_projection_rows"] = str(projection_rows_path)
        manifest["paths"]["analysis_projection_summary"] = str(projection_summary_path)

    if not args.skip_blimp:
        blimp_repeats = (
            int(args.blimp_random_control_repeats)
            if args.blimp_random_control_repeats is not None
            else int((summary.get("config") or {}).get("random_control_repeats") or 10)
        )
        blimp_alpha_grid = (
            tuple(float(alpha) for alpha in args.blimp_alpha_grid)
            if args.blimp_alpha_grid is not None
            else tuple(float(alpha) for alpha in ((summary.get("config") or {}).get("alpha_grid") or [source_selected_alpha]))
        )
        blimp = compute_blimp_transfer(
            run_dir=run_dir,
            summary=summary,
            selected_hook=selected_hook,
            steering_vector=steering_vector,
            source_selected_alpha=source_selected_alpha,
            normalize_flag=normalize_flag,
            batch_size=args.blimp_batch_size,
            blimp_config=args.blimp_config,
            random_control_repeats=blimp_repeats,
            validation_size=int(args.blimp_validation_size),
            alpha_grid=blimp_alpha_grid,
            selection_effect_fraction=float(args.blimp_selection_effect_fraction),
        )
        blimp_status_path = run_dir / f"blimp_transfer_status_{analysis_tag}.json"
        blimp_failures_path = run_dir / f"blimp_transfer_alignment_failures_{analysis_tag}.csv"
        blimp_rows_path = run_dir / f"blimp_transfer_rows_{analysis_tag}.csv"
        blimp_summary_path = run_dir / f"blimp_transfer_summary_{analysis_tag}.csv"
        blimp_random_path = run_dir / f"blimp_transfer_random_summary_{analysis_tag}.csv"
        blimp_validation_path = run_dir / f"blimp_transfer_validation_sweep_{analysis_tag}.csv"

        save_json(blimp_status_path, blimp["status"])
        save_csv(blimp["alignment_failures"], blimp_failures_path, index=False)
        save_csv(blimp["results"], blimp_rows_path, index=False)
        save_csv(blimp["summary"], blimp_summary_path, index=False)
        save_csv(blimp["random_summary"], blimp_random_path, index=False)
        save_csv(blimp["validation_sweep"], blimp_validation_path, index=False)

        manifest["paths"]["blimp_transfer_status"] = str(blimp_status_path)
        manifest["paths"]["blimp_transfer_alignment_failures"] = str(blimp_failures_path)
        manifest["paths"]["blimp_transfer_rows"] = str(blimp_rows_path)
        manifest["paths"]["blimp_transfer_summary"] = str(blimp_summary_path)
        manifest["paths"]["blimp_transfer_random_summary"] = str(blimp_random_path)
        manifest["paths"]["blimp_transfer_validation_sweep"] = str(blimp_validation_path)

    manifest_path = run_dir / f"concept_analysis_artifacts_{analysis_tag}.json"
    save_json(manifest_path, manifest)
    print(f"Wrote concept-analysis artifacts manifest: {manifest_path}")
    for name, path in manifest["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

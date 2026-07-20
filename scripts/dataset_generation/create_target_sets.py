from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import inflect
import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer

ANIMATE_QUOTA = 500
INANIMATE_QUOTA = 500
MODEL_NAME = "gpt2"
STRICT_SENSE_POLICY = "all_senses_must_qualify"

ANIMATE_LEXNAMES = {"noun.person"}
INANIMATE_LEXNAMES = {
    "noun.event",
    "noun.phenomenon",
    "noun.state",
    # "noun.act",
    "noun.artifact",
    "noun.substance",
    "noun.object",
}

PERSON_SYNSET_NAME = "person.n.01"
GROUP_SYNSET_NAME = "group.n.01"

NON_ALPHA = re.compile(r"^[a-z]+$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    return repo_root() / "dataset" / "semantic_meaningful" / "wordnet_lexname_targets_500x500.json"

# add near the other path helpers
def default_semantic_groups_path() -> Path:
    return repo_root() / "dataset" / "semantic_meaningful" / "semantic_groups.json"


# extend parse_args()
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create strict WordNet lexname-constrained target sets with exactly "
            "500 animate and 500 inanimate single-token nouns."
        )
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--semantic-groups",
        type=Path,
        default=default_semantic_groups_path(),
        help="Path to semantic_groups.json used for high-priority must-include targets.",
    )
    parser.add_argument(
        "--no-repeatability-check",
        action="store_true",
        help="Skip in-process second pass used to verify deterministic reproducibility.",
    )
    return parser.parse_args()


def tok_len(word: str, tokenizer: AutoTokenizer) -> int:
    # Keep token counting compatible with existing pipeline conventions.
    return len(tokenizer.encode(" " + word))


def normalize_lemma_name(lemma_name: str) -> str | None:
    token = lemma_name.lower().replace("_", " ").strip()
    if " " in token or "-" in token:
        return None
    if not NON_ALPHA.fullmatch(token):
        return None
    return token


def singularize_word(word: str, infl: inflect.engine) -> str | None:
    singular = infl.singular_noun(word)
    candidate = singular if isinstance(singular, str) and singular else word
    if not NON_ALPHA.fullmatch(candidate):
        return None
    return candidate.lower()


def normalize_and_singularize(lemma_name: str, infl: inflect.engine) -> str | None:
    normalized = normalize_lemma_name(lemma_name)
    if normalized is None:
        return None
    return singularize_word(normalized, infl)

# add after normalize_and_singularize()
def load_semantic_priority_targets(path: Path, infl: inflect.engine) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    animate: set[str] = set()
    inanimate: set[str] = set()
    animate_drops: set[str] = set()
    inanimate_drops: set[str] = set()
    animate_raw_count = 0
    inanimate_raw_count = 0

    for frame in payload.get("frames", []):
        for word in frame.get("animate_targets", []):
            animate_raw_count += 1
            normalized = normalize_and_singularize(word, infl)
            if normalized is None:
                animate_drops.add(word)
                continue
            animate.add(normalized)

        for word in frame.get("inanimate_targets", []):
            inanimate_raw_count += 1
            normalized = normalize_and_singularize(word, infl)
            if normalized is None:
                inanimate_drops.add(word)
                continue
            inanimate.add(normalized)

    return {
        "animate": sorted(animate),
        "inanimate": sorted(inanimate),
        "raw_counts": {
            "animate": animate_raw_count,
            "inanimate": inanimate_raw_count,
        },
        "normalization_drops": {
            "animate": sorted(animate_drops),
            "inanimate": sorted(inanimate_drops),
        },
    }


def lexname_seed_candidates(
    allowed_lexnames: set[str],
    infl: inflect.engine,
) -> tuple[set[str], int]:
    candidates: set[str] = set()
    raw_lemma_hits = 0

    for synset in wn.all_synsets(pos=wn.NOUN):
        if synset.lexname() not in allowed_lexnames:
            continue

        for lemma in synset.lemmas():
            raw_lemma_hits += 1
            candidate = normalize_and_singularize(lemma.name(), infl)
            if candidate is None:
                continue
            candidates.add(candidate)

    return candidates, raw_lemma_hits


def synset_cache_for_word(word: str) -> list[dict]:
    senses = sorted(wn.synsets(word, pos=wn.NOUN), key=lambda s: s.name())
    cached: list[dict] = []

    for synset in senses:
        closure = synset.closure(lambda s: s.hypernyms() + s.instance_hypernyms())
        closure_names = sorted({ancestor.name() for ancestor in closure})

        path_names = [
            [node.name() for node in path]
            for path in synset.hypernym_paths()
        ]

        has_person_path = any(PERSON_SYNSET_NAME in path for path in path_names)
        has_group_ancestor = GROUP_SYNSET_NAME in closure_names

        cached.append(
            {
                "name": synset.name(),
                "lexname": synset.lexname(),
                "definition": synset.definition(),
                "hypernym_paths": path_names,
                "hypernym_closure": closure_names,
                "has_person_path": has_person_path,
                "has_group_ancestor": has_group_ancestor,
            }
        )

    return cached


def lemma_count_on_sense(word: str, synset_name: str, infl: inflect.engine) -> int:
    synset = wn.synset(synset_name)
    score = 0
    for lemma in synset.lemmas():
        normalized = normalize_and_singularize(lemma.name(), infl)
        if normalized == word:
            score += int(lemma.count())
    return score


def build_profiles(words: set[str]) -> dict[str, list[dict]]:
    return {word: synset_cache_for_word(word) for word in sorted(words)}


def evaluate_animate(word: str, senses: list[dict], token_len: int) -> list[str]:
    reasons: list[str] = []

    if token_len != 1:
        reasons.append("token_len_not_one")
    if not senses:
        reasons.append("no_noun_senses")
        return reasons

    if any(s["lexname"] != "noun.person" for s in senses):
        reasons.append("non_person_lexname_present")

    if not any(bool(s["has_person_path"]) for s in senses):
        reasons.append("missing_person_path")

    if any(bool(s["has_group_ancestor"]) for s in senses):
        reasons.append("group_ancestor_present")

    return reasons


def evaluate_inanimate(word: str, senses: list[dict], token_len: int) -> list[str]:
    reasons: list[str] = []

    if token_len != 1:
        reasons.append("token_len_not_one")
    if not senses:
        reasons.append("no_noun_senses")
        return reasons

    lexnames = {s["lexname"] for s in senses}
    if "noun.person" in lexnames:
        reasons.append("contains_noun_person_sense")

    if any(lex not in INANIMATE_LEXNAMES for lex in lexnames):
        reasons.append("contains_lexname_outside_allowed_inanimate_set")

    if any(bool(s["has_group_ancestor"]) for s in senses):
        reasons.append("group_ancestor_present")

    return reasons


def rank_words(
    accepted_words: set[str],
    profiles: dict[str, list[dict]],
    infl: inflect.engine,
) -> list[dict]:
    ranked: list[dict] = []

    for word in sorted(accepted_words):
        senses = profiles[word]
        score = sum(lemma_count_on_sense(word, s["name"], infl) for s in senses)
        ranked.append(
            {
                "word": word,
                "score": int(score),
                "sense_count": len(senses),
                "sense_names": [s["name"] for s in senses],
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["word"]))
    return ranked


def top_reason_examples(reason_map: dict[str, list[str]], top_n: int = 10) -> dict[str, dict]:
    reason_counts = {reason: len(words) for reason, words in reason_map.items()}
    ordered = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    out: dict[str, dict] = {}
    for reason, count in ordered:
        examples = sorted(reason_map[reason])[:top_n]
        out[reason] = {"count": count, "examples": examples}
    return out

# add after top_reason_examples()
def filter_priority_targets(
    words: list[str],
    profiles: dict[str, list[dict]],
    tokenizer: AutoTokenizer,
    kind: str,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    valid: list[str] = []
    failed_by_word: dict[str, list[str]] = {}
    reason_map: dict[str, list[str]] = defaultdict(list)

    evaluator = evaluate_animate if kind == "animate" else evaluate_inanimate

    for word in sorted(set(words)):
        senses = profiles.get(word, [])
        token_len = tok_len(word, tokenizer)
        reasons = evaluator(word, senses, token_len)
        if reasons:
            failed_by_word[word] = reasons
            for reason in reasons:
                reason_map[reason].append(word)
        else:
            valid.append(word)

    return valid, failed_by_word, reason_map


def select_with_must_include(
    ranked: list[dict],
    must_include_words: list[str],
    quota: int,
    label: str,
) -> tuple[list[dict], dict]:
    must_include_set = set(must_include_words)
    if len(must_include_set) > quota:
        raise RuntimeError(
            f"{label} must-include set has {len(must_include_set)} items, "
            f"which exceeds quota {quota}."
        )

    ranked_by_word = {row["word"]: row for row in ranked}
    missing_from_ranked = sorted(must_include_set - set(ranked_by_word))
    if missing_from_ranked:
        raise RuntimeError(
            f"{label} valid must-include words missing from ranked pool: "
            f"{missing_from_ranked}"
        )

    baseline = ranked[:quota]
    baseline_words = {row["word"] for row in baseline}

    baseline_hits = [row for row in baseline if row["word"] in must_include_set]
    forced_additions = [
        row
        for row in ranked
        if row["word"] in must_include_set and row["word"] not in baseline_words
    ]

    selected: list[dict] = []
    selected_words: set[str] = set()

    for row in baseline_hits + forced_additions:
        if row["word"] not in selected_words:
            selected.append(row)
            selected_words.add(row["word"])

    for row in ranked:
        if row["word"] in selected_words:
            continue
        selected.append(row)
        selected_words.add(row["word"])
        if len(selected) == quota:
            break

    if len(selected) != quota:
        raise RuntimeError(
            f"{label} selection could not be topped up to quota {quota}; "
            f"got {len(selected)}."
        )

    return selected, {
        "must_include_total": len(must_include_set),
        "must_include_words": sorted(must_include_set),
        "baseline_hits": [row["word"] for row in baseline_hits],
        "forced_additions": [row["word"] for row in forced_additions],
        "generic_top_up_count": quota - len(must_include_set),
    }


def extract_targets(
    tokenizer: AutoTokenizer,
    infl: inflect.engine,
    semantic_groups_path: Path,
) -> dict:
    person_synset = wn.synset(PERSON_SYNSET_NAME)
    group_synset = wn.synset(GROUP_SYNSET_NAME)

    animate_seed, animate_raw_hits = lexname_seed_candidates(ANIMATE_LEXNAMES, infl)
    inanimate_seed, inanimate_raw_hits = lexname_seed_candidates(INANIMATE_LEXNAMES, infl)

    priority_targets = load_semantic_priority_targets(semantic_groups_path, infl)

    all_seed_words = (
        animate_seed
        | inanimate_seed
        | set(priority_targets["animate"])
        | set(priority_targets["inanimate"])
    )
    profiles = build_profiles(all_seed_words)

    animate_reasons: dict[str, list[str]] = defaultdict(list)
    inanimate_reasons: dict[str, list[str]] = defaultdict(list)

    animate_accepted: set[str] = set()
    inanimate_accepted: set[str] = set()

    for word in sorted(animate_seed):
        senses = profiles[word]
        token_len = tok_len(word, tokenizer)
        reasons = evaluate_animate(word, senses, token_len)

        if reasons:
            for reason in reasons:
                animate_reasons[reason].append(word)
        else:
            animate_accepted.add(word)

    for word in sorted(inanimate_seed):
        senses = profiles[word]
        token_len = tok_len(word, tokenizer)
        reasons = evaluate_inanimate(word, senses, token_len)

        if reasons:
            for reason in reasons:
                inanimate_reasons[reason].append(word)
        else:
            inanimate_accepted.add(word)

    valid_animate_priority, animate_priority_failures, animate_priority_reason_map = (
        filter_priority_targets(
            priority_targets["animate"],
            profiles,
            tokenizer,
            kind="animate",
        )
    )
    valid_inanimate_priority, inanimate_priority_failures, inanimate_priority_reason_map = (
        filter_priority_targets(
            priority_targets["inanimate"],
            profiles,
            tokenizer,
            kind="inanimate",
        )
    )

    animate_accepted.update(valid_animate_priority)
    inanimate_accepted.update(valid_inanimate_priority)

    overlap = animate_accepted & inanimate_accepted
    if overlap:
        animate_accepted -= overlap
        inanimate_accepted -= overlap
        valid_animate_priority = [word for word in valid_animate_priority if word not in overlap]
        valid_inanimate_priority = [word for word in valid_inanimate_priority if word not in overlap]

    animate_ranked = rank_words(animate_accepted, profiles, infl)
    inanimate_ranked = rank_words(inanimate_accepted, profiles, infl)

    if len(animate_ranked) < ANIMATE_QUOTA or len(inanimate_ranked) < INANIMATE_QUOTA:
        shortfall_report = {
            "animate_available": len(animate_ranked),
            "inanimate_available": len(inanimate_ranked),
            "animate_required": ANIMATE_QUOTA,
            "inanimate_required": INANIMATE_QUOTA,
            "animate_shortfall": max(0, ANIMATE_QUOTA - len(animate_ranked)),
            "inanimate_shortfall": max(0, INANIMATE_QUOTA - len(inanimate_ranked)),
            "top_animate_drop_reasons": top_reason_examples(animate_reasons),
            "top_inanimate_drop_reasons": top_reason_examples(inanimate_reasons),
            "animate_priority_failures": animate_priority_failures,
            "inanimate_priority_failures": inanimate_priority_failures,
        }
        raise RuntimeError(
            "Strict quotas could not be met under all-senses constraints. "
            f"Shortfall report: {json.dumps(shortfall_report, ensure_ascii=False)}"
        )

    animate_selected, animate_priority_selection = select_with_must_include(
        animate_ranked,
        valid_animate_priority,
        ANIMATE_QUOTA,
        label="animate",
    )
    inanimate_selected, inanimate_priority_selection = select_with_must_include(
        inanimate_ranked,
        valid_inanimate_priority,
        INANIMATE_QUOTA,
        label="inanimate",
    )

    animate_words = [row["word"] for row in animate_selected]
    inanimate_words = [row["word"] for row in inanimate_selected]

    assert len(animate_words) == ANIMATE_QUOTA, "Animate target count mismatch."
    assert len(inanimate_words) == INANIMATE_QUOTA, "Inanimate target count mismatch."
    assert len(set(animate_words)) == ANIMATE_QUOTA, "Animate targets contain duplicates."
    assert len(set(inanimate_words)) == INANIMATE_QUOTA, "Inanimate targets contain duplicates."
    assert set(animate_words).isdisjoint(set(inanimate_words)), "Final targets are not disjoint."

    for word in animate_words:
        assert tok_len(word, tokenizer) == 1, f"Animate word is not 1-token: {word}"
        senses = profiles[word]
        assert len(senses) > 0, f"Animate word has no noun senses: {word}"
        assert all(s["lexname"] == "noun.person" for s in senses), f"Animate lexname violation: {word}"
        assert any(bool(s["has_person_path"]) for s in senses), f"Animate person path missing: {word}"
        assert not any(bool(s["has_group_ancestor"]) for s in senses), f"Animate group ancestor hit: {word}"

    for word in inanimate_words:
        assert tok_len(word, tokenizer) == 1, f"Inanimate word is not 1-token: {word}"
        senses = profiles[word]
        assert len(senses) > 0, f"Inanimate word has no noun senses: {word}"
        assert all(
            s["lexname"] in INANIMATE_LEXNAMES for s in senses
        ), f"Inanimate lexname violation: {word}"
        assert all(
            s["lexname"] != "noun.person" for s in senses
        ), f"Inanimate noun.person violation: {word}"
        assert not any(
            bool(s["has_group_ancestor"]) for s in senses
        ), f"Inanimate group ancestor hit: {word}"

    animate_sample = []
    for word in animate_words[:10]:
        animate_sample.append(
            {
                "word": word,
                "score": next(item["score"] for item in animate_selected if item["word"] == word),
                "first_sense": profiles[word][0]["name"],
                "first_definition": profiles[word][0]["definition"],
            }
        )

    inanimate_sample = []
    for word in inanimate_words[:10]:
        inanimate_sample.append(
            {
                "word": word,
                "score": next(item["score"] for item in inanimate_selected if item["word"] == word),
                "first_sense": profiles[word][0]["name"],
                "first_definition": profiles[word][0]["definition"],
            }
        )

    return {
        "meta": {
            "script": "create_target_sets.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "WordNet via nltk.corpus.wordnet",
            "strict_policy": STRICT_SENSE_POLICY,
            "fallback_policy": "fail_fast",
            "target_counts": {"animate": ANIMATE_QUOTA, "inanimate": INANIMATE_QUOTA},
            "tokenizer": {
                "model_name": tokenizer.name_or_path,
                "one_token_check": "len(tokenizer.encode(' ' + word)) == 1",
            },
            "normalization": {
                "lowercase": True,
                "single_word_alpha_only": True,
                "singularization": "inflect.engine().singular_noun",
            },
            "lexname_constraints": {
                "animate_allowed": sorted(ANIMATE_LEXNAMES),
                "inanimate_allowed": sorted(INANIMATE_LEXNAMES),
            },
            "semantic_guards": {
                "animate_requires_person_path": PERSON_SYNSET_NAME,
                "exclude_group_ancestor": GROUP_SYNSET_NAME,
                "person_synset_reference": person_synset.name(),
                "group_synset_reference": group_synset.name(),
            },
            "selection_policy": "must_include_priority_then_ranked_top_up",
            "semantic_priority_source": str(semantic_groups_path),
            "deterministic_sort": "score_desc_then_word_asc",
        },
        "counts": {
            "phase2_seed_lemma_hits": {
                "animate": animate_raw_hits,
                "inanimate": inanimate_raw_hits,
            },
            "phase2_seed_unique_candidates": {
                "animate": len(animate_seed),
                "inanimate": len(inanimate_seed),
                "union": len(all_seed_words),
            },
            "phase3_after_strict_pre_disjoint": {
                "animate": len(animate_accepted | overlap),
                "inanimate": len(inanimate_accepted | overlap),
            },
            "phase3_disjointness": {
                "overlap_removed": len(overlap),
                "animate_after_disjoint": len(animate_accepted),
                "inanimate_after_disjoint": len(inanimate_accepted),
            },
            "phase4_semantic_priority": {
                "raw_animate_targets": priority_targets["raw_counts"]["animate"],
                "raw_inanimate_targets": priority_targets["raw_counts"]["inanimate"],
                "normalized_unique_animate_targets": len(priority_targets["animate"]),
                "normalized_unique_inanimate_targets": len(priority_targets["inanimate"]),
                "valid_animate_targets": len(valid_animate_priority),
                "valid_inanimate_targets": len(valid_inanimate_priority),
                "invalid_animate_targets": len(animate_priority_failures),
                "invalid_inanimate_targets": len(inanimate_priority_failures),
            },
            "phase5_selected": {
                "animate": len(animate_words),
                "inanimate": len(inanimate_words),
            },
        },
        "targets": {
            "animate": animate_words,
            "inanimate": inanimate_words,
        },
        "ranked_selection": {
            "animate": animate_selected,
            "inanimate": inanimate_selected,
        },
        "diagnostics": {
            "excluded_overlaps": sorted(overlap),
            "excluded_group_hits": {
                "animate": sorted(set(animate_reasons.get("group_ancestor_present", []))),
                "inanimate": sorted(set(inanimate_reasons.get("group_ancestor_present", []))),
            },
            "top_dropped_reasons": {
                "animate": top_reason_examples(animate_reasons),
                "inanimate": top_reason_examples(inanimate_reasons),
            },
            "semantic_priority": {
                "normalization_drops": priority_targets["normalization_drops"],
                "valid_must_include": {
                    "animate": valid_animate_priority,
                    "inanimate": valid_inanimate_priority,
                },
                "invalid_must_include": {
                    "animate": animate_priority_failures,
                    "inanimate": inanimate_priority_failures,
                },
                "top_invalid_reasons": {
                    "animate": top_reason_examples(animate_priority_reason_map),
                    "inanimate": top_reason_examples(inanimate_priority_reason_map),
                },
                "selection": {
                    "animate": animate_priority_selection,
                    "inanimate": inanimate_priority_selection,
                },
            },
            "manual_semantic_sanity_sample": {
                "animate": animate_sample,
                "inanimate": inanimate_sample,
            },
        },
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    infl = inflect.engine()

    artifact = extract_targets(
        tokenizer=tokenizer,
        infl=infl,
        semantic_groups_path=args.semantic_groups,
    )

    if not args.no_repeatability_check:
        repeat = extract_targets(
            tokenizer=tokenizer,
            infl=infl,
            semantic_groups_path=args.semantic_groups,
        )
        if artifact["targets"] != repeat["targets"]:
            raise AssertionError("Repeatability check failed: target ordering changed across re-runs.")
        if artifact["counts"] != repeat["counts"]:
            raise AssertionError("Repeatability check failed: metadata counts changed across re-runs.")

    save_json(args.output, artifact)

    print(f"Saved strict target artifact to: {args.output.resolve()}")
    print(f"Animate targets: {len(artifact['targets']['animate'])}")
    print(f"Inanimate targets: {len(artifact['targets']['inanimate'])}")
    print("Deterministic check: passed" if not args.no_repeatability_check else "Deterministic check: skipped")


if __name__ == "__main__":
    main()

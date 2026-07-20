from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import inflect
import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer

AGENTIVE_QUOTA = 90
INANIMATE_QUOTA = 500
MODEL_NAME = "gpt2"

AGENTIVE_LEXNAMES = {"noun.group"}
AGENTIVE_ANCESTOR_NAMES = {
    "administrative_unit.n.01",
    "agency.n.01",
    "assembly.n.04",
    "coalition.n.01",
    "committee.n.01",
    "company.n.01",
    "court.n.01",
    "department.n.01",
    "enterprise.n.02",
    "government.n.01",
    "institution.n.01",
    "organization.n.01",
    "polity.n.02",
    "union.n.01",
}
NOISY_AGENTIVE_TARGETS = {
    "acc",
    "ang",
    "armour",
    "cis",
    "dod",
    "doi",
    "dos",
    "ec",
    "elevated",
    "fps",
    "friendly",
    "hostile",
    "ic",
    "indie",
    "ins",
    "isn",
    "pac",
    "sa",
    "sc",
    "ss",
    "tc",
    "un",
    "va",
    "who",
    "eight",
    "eleven",
    "five",
    "nine",
    "agriculture",
    "detail",
    "duo",
    "energy",
    "faith",
    "kindergarten",
    "law",
    "offence",
    "offense",
    "preschool",
    "rank",
    "religion",
    "sec",
    "tech",
    "trio",
}
NOISY_INANIMATE_TARGETS = {
    "hr",
    "yr",
}
ABSTRACT_INANIMATE_LEXNAMES = {
    "noun.act",
    "noun.attribute",
    "noun.cognition",
    "noun.communication",
    "noun.event",
    "noun.feeling",
    "noun.motive",
    "noun.phenomenon",
    "noun.process",
    "noun.relation",
    "noun.shape",
    "noun.state",
    "noun.time",
}
PHYSICAL_LEXNAMES = {
    "noun.animal",
    "noun.artifact",
    "noun.body",
    "noun.food",
    "noun.location",
    "noun.object",
    "noun.plant",
    "noun.possession",
    "noun.substance",
}
PERSON_SYNSET_NAME = "person.n.01"
GROUP_SYNSET_NAME = "group.n.01"

NON_ALPHA = re.compile(r"^[A-Za-z]+$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    return (
        repo_root()
        / "dataset"
        / "semantic_meaningful"
        / "abstract_agency_targets_strict_90x500.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create target sets for abstract agentive entities versus abstract "
            "non-agentive inanimate entities."
        )
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--agentive-quota",
        type=int,
        default=AGENTIVE_QUOTA,
        help="Number of abstract agentive targets to select.",
    )
    parser.add_argument(
        "--inanimate-quota",
        type=int,
        default=INANIMATE_QUOTA,
        help="Number of abstract inanimate targets to select.",
    )
    parser.add_argument(
        "--no-repeatability-check",
        action="store_true",
        help="Skip in-process second pass used to verify deterministic reproducibility.",
    )
    return parser.parse_args()


def tok_len(word: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(" " + word))


def normalize_lemma_name(lemma_name: str, infl: inflect.engine) -> str | None:
    token = lemma_name.replace("_", " ").strip()
    if " " in token or "-" in token:
        return None
    if not NON_ALPHA.fullmatch(token):
        return None
    token = token.lower()
    singular = infl.singular_noun(token)
    if isinstance(singular, str) and singular:
        token = singular
    if len(token) <= 2:
        return None
    return token


def synset_profile(word: str) -> list[dict]:
    senses = list(wn.synsets(word, pos=wn.NOUN))
    profile: list[dict] = []
    for index, synset in enumerate(senses):
        closure = synset.closure(lambda s: s.hypernyms() + s.instance_hypernyms())
        closure_names = sorted({ancestor.name() for ancestor in closure})
        path_names = [[node.name() for node in path] for path in synset.hypernym_paths()]
        profile.append(
            {
                "name": synset.name(),
                "lexname": synset.lexname(),
                "definition": synset.definition(),
                "is_first_sense": index == 0,
                "lemma_count": sum(int(lemma.count()) for lemma in synset.lemmas()),
                "has_person_path": any(PERSON_SYNSET_NAME in path for path in path_names),
                "has_group_ancestor": GROUP_SYNSET_NAME in closure_names,
                "has_agentive_ancestor": any(
                    any(ancestor in path for ancestor in AGENTIVE_ANCESTOR_NAMES)
                    for path in path_names
                ),
            }
        )
    return profile


def lemma_candidates(lexnames: set[str], infl: inflect.engine) -> tuple[set[str], int]:
    candidates: set[str] = set()
    raw_hits = 0
    for synset in wn.all_synsets(pos=wn.NOUN):
        if synset.lexname() not in lexnames:
            continue
        for lemma in synset.lemmas():
            raw_hits += 1
            normalized = normalize_lemma_name(lemma.name(), infl)
            if normalized is not None:
                candidates.add(normalized)
    return candidates, raw_hits


def agentive_seed_candidates(infl: inflect.engine) -> tuple[set[str], int]:
    candidates: set[str] = set()
    raw_lemma_hits = 0
    for synset in wn.all_synsets(pos=wn.NOUN):
        path_names = [[node.name() for node in path] for path in synset.hypernym_paths()]
        if not any(
            any(ancestor in path for ancestor in AGENTIVE_ANCESTOR_NAMES)
            for path in path_names
        ):
            continue
        for lemma in synset.lemmas():
            raw_lemma_hits += 1
            normalized = normalize_lemma_name(lemma.name(), infl)
            if normalized is not None:
                candidates.add(normalized)
    return candidates, raw_lemma_hits


def evaluate_agentive(word: str, senses: list[dict], token_len: int) -> list[str]:
    reasons: list[str] = []
    if token_len != 1:
        reasons.append("token_len_not_one")
    if not senses:
        reasons.append("no_noun_senses")
        return reasons

    lexnames = {sense["lexname"] for sense in senses}
    if word in NOISY_AGENTIVE_TARGETS:
        reasons.append("manual_noise_blocklist")
    if not lexnames & AGENTIVE_LEXNAMES:
        reasons.append("missing_group_sense")
    if not any(bool(sense["has_agentive_ancestor"]) for sense in senses):
        reasons.append("missing_agentive_ancestor")
    if lexnames & PHYSICAL_LEXNAMES:
        reasons.append("physical_lexname_present")
    if "noun.person" in lexnames:
        reasons.append("person_lexname_present")
    if any(bool(sense["has_person_path"]) for sense in senses):
        reasons.append("person_path_present")
    if not any(bool(sense["has_group_ancestor"]) for sense in senses):
        reasons.append("missing_group_ancestor")
    return reasons


def evaluate_abstract_inanimate(word: str, senses: list[dict], token_len: int) -> list[str]:
    reasons: list[str] = []
    if token_len != 1:
        reasons.append("token_len_not_one")
    if not senses:
        reasons.append("no_noun_senses")
        return reasons

    lexnames = {sense["lexname"] for sense in senses}
    if word in NOISY_INANIMATE_TARGETS:
        reasons.append("manual_noise_blocklist")
    if "noun.person" in lexnames:
        reasons.append("person_lexname_present")
    if "noun.group" in lexnames:
        reasons.append("group_lexname_present")
    if lexnames & PHYSICAL_LEXNAMES:
        reasons.append("physical_lexname_present")
    if not lexnames <= ABSTRACT_INANIMATE_LEXNAMES:
        reasons.append("non_abstract_inanimate_lexname_present")
    if any(bool(sense["has_person_path"]) for sense in senses):
        reasons.append("person_path_present")
    if any(bool(sense["has_group_ancestor"]) for sense in senses):
        reasons.append("group_ancestor_present")
    return reasons


def score_word(word: str, senses: list[dict]) -> int:
    return sum(int(sense["lemma_count"]) for sense in senses)


def agentive_priority(senses: list[dict]) -> int:
    if any(
        bool(sense["is_first_sense"]) and bool(sense["has_agentive_ancestor"])
        for sense in senses
    ):
        return 2
    if any(bool(sense["has_agentive_ancestor"]) for sense in senses):
        return 1
    return 0


def top_reason_examples(reason_map: dict[str, list[str]], top_n: int = 10) -> dict[str, dict]:
    ordered = sorted(reason_map.items(), key=lambda item: (-len(item[1]), item[0]))
    return {
        reason: {"count": len(words), "examples": sorted(words)[:top_n]}
        for reason, words in ordered
    }


def select_targets(
    candidates: Iterable[str],
    profiles: dict[str, list[dict]],
    tokenizer: AutoTokenizer,
    quota: int,
    evaluator,
    priority_fn=None,
) -> tuple[list[dict], dict[str, list[str]]]:
    accepted: list[dict] = []
    reasons_by_kind: dict[str, list[str]] = defaultdict(list)
    for word in sorted(candidates):
        senses = profiles[word]
        reasons = evaluator(word, senses, tok_len(word, tokenizer))
        priority = int(priority_fn(senses)) if priority_fn is not None else 0
        if priority_fn is not None and priority < 1:
            reasons.append("below_minimum_confidence_tier")
        if reasons:
            for reason in reasons:
                reasons_by_kind[reason].append(word)
            continue
        accepted.append(
            {
                "word": word,
                "score": score_word(word, senses),
                "priority": priority,
                "sense_count": len(senses),
                "sense_names": [sense["name"] for sense in senses],
            }
        )
    accepted.sort(key=lambda item: (-item["priority"], -item["score"], item["word"]))
    return accepted[:quota], reasons_by_kind


def extract_targets(
    tokenizer: AutoTokenizer,
    infl: inflect.engine,
    agentive_quota: int,
    inanimate_quota: int,
) -> dict:
    agentive_seed, agentive_raw_hits = lemma_candidates(AGENTIVE_LEXNAMES, infl)
    inanimate_seed, inanimate_raw_hits = lemma_candidates(ABSTRACT_INANIMATE_LEXNAMES, infl)
    all_words = agentive_seed | inanimate_seed
    profiles = {word: synset_profile(word) for word in sorted(all_words)}

    agentive_selected, agentive_reasons = select_targets(
        agentive_seed,
        profiles,
        tokenizer,
        agentive_quota,
        evaluate_agentive,
        priority_fn=agentive_priority,
    )
    agentive_words = [row["word"] for row in agentive_selected]
    agentive_word_set = set(agentive_words)

    inanimate_candidates = sorted(word for word in inanimate_seed if word not in agentive_word_set)
    inanimate_selected, inanimate_reasons = select_targets(
        inanimate_candidates,
        profiles,
        tokenizer,
        inanimate_quota,
        evaluate_abstract_inanimate,
    )
    inanimate_words = [row["word"] for row in inanimate_selected]

    if len(agentive_words) != agentive_quota or len(inanimate_words) != inanimate_quota:
        raise RuntimeError(
            "Could not meet abstract agency target quotas. "
            + json.dumps(
                {
                    "agentive_available": len(agentive_selected),
                    "agentive_required": agentive_quota,
                    "inanimate_available": len(inanimate_selected),
                    "inanimate_required": inanimate_quota,
                    "top_agentive_drop_reasons": top_reason_examples(agentive_reasons),
                    "top_inanimate_drop_reasons": top_reason_examples(inanimate_reasons),
                },
                ensure_ascii=False,
            )
        )

    assert len(set(agentive_words)) == agentive_quota
    assert len(set(inanimate_words)) == inanimate_quota
    assert set(agentive_words).isdisjoint(inanimate_words)

    return {
        "meta": {
            "script": "create_abstract_agency_target_sets.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "WordNet via nltk.corpus.wordnet",
            "target_source": "abstract_agency",
            "target_counts": {
                "animate": agentive_quota,
                "inanimate": inanimate_quota,
            },
            "semantic_interpretation": {
                "animate": "abstract agentive entities, including bodiless institutional agents",
                "inanimate": "abstract non-agentive inanimate entities",
            },
            "template_constraint": "Targets are common nouns compatible with prompts ending in 'by the'.",
            "proper_name_policy": "raw proper names excluded",
            "agentive_selection_priority": (
                "first-sense institutional/organizational agent, then secondary-sense "
                "institutional/organizational agent; generic noun.group top-up is rejected"
            ),
            "normalization": {
                "lowercase": True,
                "single_word_alpha_only": True,
                "singularization": "inflect.engine().singular_noun",
                "minimum_length": 3,
            },
            "manual_noise_blocklist": {
                "animate": sorted(NOISY_AGENTIVE_TARGETS),
                "inanimate": sorted(NOISY_INANIMATE_TARGETS),
            },
            "tokenizer": {
                "model_name": tokenizer.name_or_path,
                "one_token_check": "len(tokenizer.encode(' ' + word)) == 1",
            },
            "lexname_constraints": {
                "agentive_seed": sorted(AGENTIVE_LEXNAMES),
                "agentive_seed_ancestors": sorted(AGENTIVE_ANCESTOR_NAMES),
                "abstract_inanimate_seed": sorted(ABSTRACT_INANIMATE_LEXNAMES),
                "physical_excluded": sorted(PHYSICAL_LEXNAMES),
            },
        },
        "counts": {
            "seed_lemma_hits": {
                "animate": agentive_raw_hits,
                "inanimate": inanimate_raw_hits,
            },
            "seed_unique_candidates": {
                "animate": len(agentive_seed),
                "inanimate": len(inanimate_seed),
                "union": len(all_words),
            },
            "selected": {
                "animate": len(agentive_words),
                "inanimate": len(inanimate_words),
            },
        },
        "targets": {
            "animate": agentive_words,
            "inanimate": inanimate_words,
        },
        "ranked_selection": {
            "animate": agentive_selected,
            "inanimate": inanimate_selected,
        },
        "diagnostics": {
            "top_dropped_reasons": {
                "animate": top_reason_examples(agentive_reasons),
                "inanimate": top_reason_examples(inanimate_reasons),
            },
        },
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    infl = inflect.engine()
    artifact = extract_targets(
        tokenizer=tokenizer,
        infl=infl,
        agentive_quota=args.agentive_quota,
        inanimate_quota=args.inanimate_quota,
    )
    if not args.no_repeatability_check:
        repeat = extract_targets(
            tokenizer=tokenizer,
            infl=infl,
            agentive_quota=args.agentive_quota,
            inanimate_quota=args.inanimate_quota,
        )
        if artifact["targets"] != repeat["targets"]:
            raise AssertionError("Repeatability check failed: target ordering changed.")
        if artifact["counts"] != repeat["counts"]:
            raise AssertionError("Repeatability check failed: metadata counts changed.")
    save_json(args.output, artifact)


if __name__ == "__main__":
    main()

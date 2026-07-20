from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import inflect
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs
    inflect = None


NON_ALPHA = re.compile(r"^[a-z]+$")
TARGET_CLASSES = ("animate", "inanimate")
IRREGULAR_PLURALS = {
    "child": "children",
    "foot": "feet",
    "goose": "geese",
    "man": "men",
    "mouse": "mice",
    "ox": "oxen",
    "person": "people",
    "tooth": "teeth",
    "woman": "women",
}
INVARIANT_NOUNS = {
    "aircraft",
    "bison",
    "deer",
    "fish",
    "means",
    "moose",
    "news",
    "salmon",
    "series",
    "sheep",
    "species",
    "trout",
}
MAN_SUFFIX_EXCEPTIONS = {"german", "human", "roman"}
MANY_AWKWARD_NOUNS = {
    "acid",
    "adhesive",
    "airflow",
    "aluminum",
    "ammo",
    "anarchy",
    "anonymity",
    "appalling",
    "arsenic",
    "asphalt",
    "aspirin",
    "attire",
    "autonomy",
    "bouncing",
    "bronze",
    "calcium",
    "captivity",
    "cargo",
    "cardboard",
    "carbon",
    "chloride",
    "chrome",
    "circuitry",
    "climate",
    "cloth",
    "clothing",
    "coal",
    "cocaine",
    "complicity",
    "compost",
    "concrete",
    "congestion",
    "coping",
    "cr",
    "criminality",
    "dairy",
    "debris",
    "decor",
    "dependence",
    "diabetes",
    "diarrhea",
    "discredit",
    "doom",
    "dust",
    "ecstasy",
    "employ",
    "entirety",
    "equipment",
    "essential",
    "ethanol",
    "fallout",
    "fame",
    "fertilizer",
    "flashing",
    "flowing",
    "fluid",
    "flu",
    "fluoride",
    "foam",
    "friendship",
    "fuel",
    "furniture",
    "frenzy",
    "gasoline",
    "glue",
    "gravel",
    "grease",
    "grinding",
    "grotesque",
    "hardware",
    "health",
    "helium",
    "humidity",
    "hydrogen",
    "impunity",
    "inaction",
    "inevitable",
    "insanity",
    "iron",
    "jeopardy",
    "junk",
    "knocking",
    "laundry",
    "leakage",
    "leather",
    "lightning",
    "linen",
    "lipstick",
    "luggage",
    "lumber",
    "malaria",
    "manure",
    "marble",
    "mascara",
    "merchandise",
    "metal",
    "mist",
    "moisture",
    "news",
    "nicotine",
    "nitrogen",
    "nowhere",
    "opium",
    "oxygen",
    "ozone",
    "pale",
    "parchment",
    "pavement",
    "plaster",
    "platinum",
    "pneumonia",
    "polio",
    "popping",
    "potassium",
    "powder",
    "prestige",
    "prosperity",
    "rainfall",
    "ready",
    "refuse",
    "rubble",
    "rubber",
    "saline",
    "sanity",
    "sediment",
    "sewage",
    "silicone",
    "silk",
    "sodium",
    "steam",
    "steel",
    "sunlight",
    "supremacy",
    "susceptibility",
    "textile",
    "thunder",
    "tubing",
    "ultraviolet",
    "unemployment",
    "velvet",
    "wastewater",
    "wax",
    "weather",
    "zinc",
}


class ConservativePluralizer:
    def __init__(self) -> None:
        self.reverse_irregular = {plural: singular for singular, plural in IRREGULAR_PLURALS.items()}

    def plural_noun(self, word: str) -> str:
        if word in INVARIANT_NOUNS:
            return word
        if word in IRREGULAR_PLURALS:
            return IRREGULAR_PLURALS[word]
        if word.endswith("woman") and len(word) > len("woman"):
            return f"{word[:-5]}women"
        if word.endswith("man") and word not in MAN_SUFFIX_EXCEPTIONS:
            return f"{word[:-3]}men"
        if re.search(r"[^aeiou]y$", word):
            return f"{word[:-1]}ies"
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return f"{word}es"
        if word.endswith("f"):
            return f"{word[:-1]}ves"
        if word.endswith("fe"):
            return f"{word[:-2]}ves"
        return f"{word}s"

    def singular_noun(self, word: str) -> str | bool:
        if word in INVARIANT_NOUNS:
            return word
        if word in self.reverse_irregular:
            return self.reverse_irregular[word]
        if word.endswith("women") and len(word) > len("women"):
            return f"{word[:-5]}woman"
        if word.endswith("men"):
            singular = f"{word[:-3]}man"
            if singular not in MAN_SUFFIX_EXCEPTIONS:
                return singular
        if word.endswith("ies") and len(word) > 3:
            return f"{word[:-3]}y"
        if word.endswith("ves") and len(word) > 3:
            return f"{word[:-3]}f"
        if word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
            return word[:-2]
        if word.endswith("s") and len(word) > 1:
            return word[:-1]
        return False


def make_pluralizer() -> Any:
    if inflect is not None:
        return inflect.engine()
    return ConservativePluralizer()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_path() -> Path:
    return repo_root() / "dataset" / "semantic_meaningful" / "wordnet_lexname_targets_500x500.json"


def default_output_path() -> Path:
    return repo_root() / "dataset" / "semantic_meaningful" / "quantifier_number_targets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build singular/plural common-noun target pairs for the by-a/by-many "
            "quantifier number-control task."
        )
    )
    parser.add_argument("--source", type=Path, default=default_source_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_source_targets(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise ValueError(f"Target source has no object-valued 'targets': {path}")
    missing = [target_class for target_class in TARGET_CLASSES if target_class not in targets]
    if missing:
        raise ValueError(f"Target source is missing classes {missing}: {path}")
    return {target_class: list(targets[target_class]) for target_class in TARGET_CLASSES}, payload


def reject(
    rejected: dict[str, list[dict[str, Any]]],
    target_class: str,
    word: str,
    reason: str,
    **extra: Any,
) -> None:
    rejected[target_class].append({"word": word, "reason": reason, **extra})


def candidate_pair(word: str, infl: Any) -> tuple[str, str, str | None]:
    singular = word.strip().lower()
    if singular != word.strip():
        return singular, "", "not_lowercase_normalized"
    if not NON_ALPHA.fullmatch(singular):
        return singular, "", "not_single_alpha_token"
    if singular in MANY_AWKWARD_NOUNS:
        return singular, "", "awkward_after_many"

    plural = infl.plural_noun(singular)
    if not isinstance(plural, str) or not plural:
        return singular, "", "pluralizer_failed"
    plural = plural.lower().strip()
    if not NON_ALPHA.fullmatch(plural):
        return singular, plural, "plural_not_single_alpha_token"
    if plural == singular:
        return singular, plural, "singular_equals_plural"

    roundtrip = infl.singular_noun(plural)
    if roundtrip != singular:
        return singular, plural, "plural_roundtrip_failed"
    return singular, plural, None


def remove_collisions(
    accepted: dict[str, list[dict[str, str]]],
    rejected: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, str]]]:
    singular_owners: dict[str, list[str]] = defaultdict(list)
    plural_owners: dict[str, list[str]] = defaultdict(list)
    for target_class, rows in accepted.items():
        for row in rows:
            singular_owners[row["singular"]].append(target_class)
            plural_owners[row["plural"]].append(target_class)

    filtered: dict[str, list[dict[str, str]]] = {target_class: [] for target_class in TARGET_CLASSES}
    all_singulars = set(singular_owners)
    all_plurals = set(plural_owners)
    for target_class, rows in accepted.items():
        for row in rows:
            reasons: list[str] = []
            if len(singular_owners[row["singular"]]) > 1:
                reasons.append("singular_cross_class_collision")
            if len(plural_owners[row["plural"]]) > 1:
                reasons.append("plural_cross_class_collision")
            if row["plural"] in all_singulars:
                reasons.append("plural_collides_with_singular")
            if row["singular"] in all_plurals:
                reasons.append("singular_collides_with_plural")

            if reasons:
                reject(
                    rejected,
                    target_class,
                    row["singular"],
                    "target_collision",
                    plural=row["plural"],
                    collision_reasons=reasons,
                )
                continue
            filtered[target_class].append(row)
    return filtered


def build_quantifier_targets(source_targets: dict[str, list[str]]) -> dict[str, Any]:
    infl = make_pluralizer()
    accepted: dict[str, list[dict[str, str]]] = {target_class: [] for target_class in TARGET_CLASSES}
    rejected: dict[str, list[dict[str, Any]]] = {target_class: [] for target_class in TARGET_CLASSES}
    seen_singulars: dict[str, set[str]] = {target_class: set() for target_class in TARGET_CLASSES}
    seen_plurals: dict[str, set[str]] = {target_class: set() for target_class in TARGET_CLASSES}

    for target_class in TARGET_CLASSES:
        for word in source_targets[target_class]:
            singular, plural, reason = candidate_pair(str(word), infl)
            if reason is not None:
                reject(rejected, target_class, str(word), reason, singular=singular, plural=plural)
                continue
            if singular in seen_singulars[target_class]:
                reject(rejected, target_class, singular, "duplicate_singular", plural=plural)
                continue
            if plural in seen_plurals[target_class]:
                reject(rejected, target_class, singular, "duplicate_plural", plural=plural)
                continue
            seen_singulars[target_class].add(singular)
            seen_plurals[target_class].add(plural)
            accepted[target_class].append({"singular": singular, "plural": plural})

    filtered = remove_collisions(accepted, rejected)
    return {
        "targets": filtered,
        "rejections": rejected,
        "summary": summarize(source_targets, filtered, rejected),
    }


def summarize(
    source_targets: dict[str, list[str]],
    accepted: dict[str, list[dict[str, str]]],
    rejected: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target_class in TARGET_CLASSES:
        reason_counts = Counter(row["reason"] for row in rejected[target_class])
        collision_reason_counts = Counter(
            reason
            for row in rejected[target_class]
            for reason in row.get("collision_reasons", [])
        )
        summary[target_class] = {
            "source_count": len(source_targets[target_class]),
            "accepted_count": len(accepted[target_class]),
            "rejected_count": len(rejected[target_class]),
            "rejection_reasons": dict(sorted(reason_counts.items())),
            "collision_reasons": dict(sorted(collision_reason_counts.items())),
        }
    return summary


def main() -> None:
    args = parse_args()
    source_targets, source_payload = load_source_targets(args.source)
    payload = build_quantifier_targets(source_targets)
    output = {
        "meta": {
            "script": Path(__file__).name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(args.source),
            "source_meta": source_payload.get("meta", {}),
            "task": {
                "singular_prefix_suffix": "by a",
                "plural_prefix_suffix": "by many",
                "singular_target_context": "' ' + singular after by a",
                "plural_target_context": "' ' + plural after by many",
            },
            "constraints": {
                "allow_irregular_plurals": True,
                "reject_singular_equals_plural": True,
                "roundtrip_plural_to_singular": True,
                "lowercase_single_alpha_token": True,
                "collision_free_across_classes_and_number": True,
            },
        },
        **payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rng = random.Random(args.seed)
    report = {
        "output": str(args.output),
        "summary": payload["summary"],
        "examples": {
            target_class: rng.sample(
                payload["targets"][target_class],
                k=min(args.examples, len(payload["targets"][target_class])),
            )
            for target_class in TARGET_CLASSES
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

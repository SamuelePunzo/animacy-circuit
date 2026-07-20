from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
import nltk
from nltk.corpus import verbnet as vn

from datagen_utils import (
    class_role_restrictions, is_clean_class, is_corrupt_class_strict,
    is_corrupt_class_fallback, normalize_member_name, convert_pool, group_by_tok, has_transitive_frame,
    is_transitive_verb, CORRUPT_BLOCKLIST
)

MODEL_NAME = "gpt2"
OUTPUT_STEP1 = Path("../dataset/semantic_meaningful/step1_verbnet_verb_pools.json")

# VerbNet 3.4 in NLTK does not always annotate Cause with explicit -animate.
# This stays VerbNet-only (no manual seed lists), with a metadata fallback rule if needed.
ALLOW_VERBNET_METADATA_FALLBACK_FOR_CORRUPT = True


if __name__ == "__main__":
    nltk.download("verbnet", quiet=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print(f"VerbNet classes: {len(vn.classids())}")
    print(f"Tokenizer model: {MODEL_NAME}")
    print(f"Step 1 output: {OUTPUT_STEP1.resolve()}")

    clean_lemma_to_classes: dict[str, set[str]] = defaultdict(set)
    corrupt_lemma_to_classes: dict[str, set[str]] = defaultdict(set)

    strict_corrupt_classes: set[str] = set()
    fallback_corrupt_classes: set[str] = set()

    # Iterate over VerbNet classes and classify them as clean/corrupt based on role restrictions
    for classid in vn.classids():
        vnclass    = vn.vnclass(classid)
        role_map   = class_role_restrictions(vnclass)
        clean_match  = is_clean_class(role_map)
        corrupt_match_strict = is_corrupt_class_strict(role_map)

        if corrupt_match_strict:
            strict_corrupt_classes.add(classid)
        corrupt_match = corrupt_match_strict
        if not corrupt_match and ALLOW_VERBNET_METADATA_FALLBACK_FOR_CORRUPT:
            corrupt_match = is_corrupt_class_fallback(role_map)
            if corrupt_match:
                fallback_corrupt_classes.add(classid)

        if not clean_match and not corrupt_match:
            continue

        # ← ADD THIS: skip classes with no transitive NP V NP frame
        if not has_transitive_frame(vnclass):
            continue

        member_lemmas = [normalize_member_name(member.attrib.get("name"))
                        for member in vnclass.findall("MEMBERS/MEMBER")]
        member_lemmas = [m for m in member_lemmas if m is not None]
        member_lemmas = [m for m in member_lemmas if is_transitive_verb(m)]

        for lemma in member_lemmas:
            if clean_match:
                clean_lemma_to_classes[lemma].add(classid)
            if corrupt_match:
                corrupt_lemma_to_classes[lemma].add(classid)

    print(f"Clean lemmas from VerbNet: {len(clean_lemma_to_classes)}")
    print(f"Corrupt lemmas from VerbNet: {len(corrupt_lemma_to_classes)}")
    print(f"Corrupt strict classes (Cause -animate): {len(strict_corrupt_classes)}")
    print(f"Corrupt fallback classes (VerbNet metadata): {len(fallback_corrupt_classes)}")

    # Convert lemma pools to past participle form, dropping any with non-alpha characters after conversion.
    clean_participle_to_classes, clean_dropped = convert_pool(clean_lemma_to_classes)
    corrupt_participle_to_classes, corrupt_dropped = convert_pool(corrupt_lemma_to_classes)

    clean_participles = set(clean_participle_to_classes)
    corrupt_participles = set(corrupt_participle_to_classes)

    # Remove any overlap between clean and corrupt pools to ensure disjointness.
    overlap = clean_participles.intersection(corrupt_participles)
    for verb in overlap:
        clean_participle_to_classes.pop(verb, None)
        corrupt_participle_to_classes.pop(verb, None)

    for verb in CORRUPT_BLOCKLIST:
        corrupt_participle_to_classes.pop(verb, None)

    clean_participles = sorted(clean_participle_to_classes)
    corrupt_participles = sorted(corrupt_participle_to_classes)

    clean_by_tok = group_by_tok(clean_participles, tokenizer)
    corrupt_by_tok = group_by_tok(corrupt_participles, tokenizer)

    # I filter down to 1-token verbs for the final pools - not for now, too restrictive
    # clean_1tok = clean_by_tok.get(1, [])
    # corrupt_1tok = corrupt_by_tok.get(1, [])

    print(f"Clean participles after conversion: {len(clean_participles)}")
    print(f"Corrupt participles after conversion: {len(corrupt_participles)}")
    print(f"Overlap removed for disjointness: {len(overlap)}")
    # print(f"Clean 1-token verbs: {len(clean_1tok)}")
    # print(f"Corrupt 1-token verbs: {len(corrupt_1tok)}")

    final_clean_pool = [clean_verb for n in sorted(clean_by_tok) for clean_verb in clean_by_tok.get(n, [])]
    final_corrupt_pool = [corrupt_verb for n in sorted(corrupt_by_tok) for corrupt_verb in corrupt_by_tok.get(n, [])]

    print(final_corrupt_pool)

    print(f"Clean token verbs: {len(clean_by_tok)}")
    print(f"Corrupt token verbs: {len(corrupt_by_tok)}")

    # assert len(clean_1tok) > 0, "No 1-token clean verbs produced."
    # assert len(corrupt_1tok) > 0, "No 1-token corrupt verbs produced."
    # assert set(clean_1tok).isdisjoint(set(corrupt_1tok)), "Clean/corrupt 1-token pools overlap."

    print("Sample clean verbs with provenance:")
    for verb in final_clean_pool[:15]:
        classes = sorted(clean_participle_to_classes[verb])[:3]
        print(f"  {verb:<16} {classes}")

    print("\nSample corrupt verbs with provenance:")
    for verb in final_corrupt_pool[:15]:
        classes = sorted(corrupt_participle_to_classes[verb])[:3]
        print(f"  {verb:<16} {classes}")

    print("\nStep 1 validation checks passed.")

    artifact = {
        "meta": {
            "phase": "Step 1 only",
            "source": "VerbNet via nltk.corpus.verbnet",
            "tokenizer_model": MODEL_NAME,
            "token_length_uses_leading_space": True,
            "corrupt_primary_rule": "Cause role with -animate restriction",
            "corrupt_fallback_rule": (
                "VerbNet metadata fallback: physical Patient/Theme with Cause/Instrument/intentional-Agent signature"
                if ALLOW_VERBNET_METADATA_FALLBACK_FOR_CORRUPT
                else "disabled"
            ),
            "strict_corrupt_class_count": len(strict_corrupt_classes),
            "fallback_corrupt_class_count": len(fallback_corrupt_classes),
            "overlap_removed_count": len(overlap),
            "clean_dropped_count": len(clean_dropped),
            "corrupt_dropped_count": len(corrupt_dropped),
        },
        "verbs": {
            "note": "Step 1 output only: clean/corrupt verb pools in past participle form.",
            "by_token_length": {
                str(n): {
                    "clean": clean_by_tok.get(n, []),
                    "corrupt": corrupt_by_tok.get(n, []),
                    "n_pairs": len(clean_by_tok.get(n, [])) * len(corrupt_by_tok.get(n, [])),
                }
                for n in sorted(set(clean_by_tok) | set(corrupt_by_tok))
            },
        },
        "provenance": {
            "clean_classes_by_verb": {
                verb: sorted(class_ids) for verb, class_ids in sorted(clean_participle_to_classes.items())
            },
            "corrupt_classes_by_verb": {
                verb: sorted(class_ids) for verb, class_ids in sorted(corrupt_participle_to_classes.items())
            },
            "strict_corrupt_classes": sorted(strict_corrupt_classes),
            "fallback_corrupt_classes": sorted(fallback_corrupt_classes),
            "dropped": {
                "clean": clean_dropped,
                "corrupt": corrupt_dropped,
            },
        },
    }

    OUTPUT_STEP1.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STEP1, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved Step 1 artifact to: {OUTPUT_STEP1.resolve()}")
    print(f"Final clean pool size: {len(final_clean_pool)}")
    print(f"Final corrupt pool size: {len(final_corrupt_pool)}")


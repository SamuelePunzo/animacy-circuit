from difflib import SequenceMatcher
from itertools import product
import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

from datagen_utils import (
    representative_class, domain_label, lexical_similarity, tok_len
)

OUTPUT_STEP3 = Path("../dataset/semantic_meaningful/step3_verb_pairs_by_domain.json")

MAX_PAIRS_PER_DOMAIN = 8
MIN_PAIRS_PER_DOMAIN = 4
MAX_CORRUPT_VERB_APPEARANCES = 15
MAX_CLEAN_VERB_PER_DOMAIN = 3

if __name__ == "__main__":
    step1_artifact = json.load(open("../dataset/semantic_meaningful/step1_verbnet_verb_pools.json", "r", encoding="utf-8"))

    MODEL_NAME = step1_artifact["meta"]["tokenizer_model"]
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    clean_classes_by_verb   = step1_artifact["provenance"]["clean_classes_by_verb"]
    corrupt_classes_by_verb = step1_artifact["provenance"]["corrupt_classes_by_verb"]

    verbs       = step1_artifact["verbs"]["by_token_length"]
    clean_all   = [v for bucket in verbs.values() for v in bucket["clean"]]
    corrupt_all = [v for bucket in verbs.values() for v in bucket["corrupt"]]

    clean_rep_class  = {v: representative_class(clean_classes_by_verb.get(v, [])) for v in clean_all}
    corrupt_rep_class = {v: representative_class(corrupt_classes_by_verb.get(v, [])) for v in corrupt_all}

    candidate_rows = []
    for clean_v, corrupt_v in product(clean_all, corrupt_all):
        if tok_len(clean_v, tokenizer) != tok_len(corrupt_v, tokenizer):
            continue

        cls = clean_rep_class.get(clean_v)
        if cls is None:
            continue
        domain = domain_label(cls)

        candidate_rows.append({
            "domain": domain,
            "clean_verb": clean_v,
            "corrupt_verb": corrupt_v,
            "clean_class": cls,
            "corrupt_class": corrupt_rep_class.get(corrupt_v),
            "lexical_similarity": round(lexical_similarity(clean_v, corrupt_v), 4),
        })

    domain_rows: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        domain_rows[row["domain"]].append(row)

    for rows in domain_rows.values():
        rows.sort(key=lambda r: (r["lexical_similarity"], r["clean_verb"], r["corrupt_verb"]))

    domain_pairs: dict[str, list[list[str]]] = {}
    domain_details: dict[str, list[dict]] = {}

    corrupt_verb_count: dict[str, int] = defaultdict(int)

    for domain, rows in sorted(domain_rows.items()):
        clean_verb_count_in_domain: dict[str, int] = defaultdict(int)
        filtered = []
        for r in rows:
            if corrupt_verb_count[r["corrupt_verb"]] >= MAX_CORRUPT_VERB_APPEARANCES:
                continue
            if clean_verb_count_in_domain[r["clean_verb"]] >= MAX_CLEAN_VERB_PER_DOMAIN:
                continue
            filtered.append(r)
            corrupt_verb_count[r["corrupt_verb"]] += 1
            clean_verb_count_in_domain[r["clean_verb"]] += 1

        kept = filtered[:MAX_PAIRS_PER_DOMAIN]
        if len(kept) < MIN_PAIRS_PER_DOMAIN:
            continue
        domain_pairs[domain]   = [[r["clean_verb"], r["corrupt_verb"]] for r in kept]
        domain_details[domain] = kept

    assert len(domain_pairs) > 0, "No Step 3 domains produced. Inspect Step 1 pools."
    assert all(len(pairs) <= MAX_PAIRS_PER_DOMAIN for pairs in domain_pairs.values())

    step3_artifact = {
        "meta": {
            "phase": "Step 3 only",
            "source": "Derived from Step 1 VerbNet provenance",
            "tokenizer_model": MODEL_NAME,
            "token_length_uses_leading_space": True,
            "derivation": {
                "domain_id": "VerbNet class name prefix of representative clean class",
                "max_pairs_per_domain": MAX_PAIRS_PER_DOMAIN,
                "min_pairs_per_domain": MIN_PAIRS_PER_DOMAIN,
                "max_corrupt_verb_appearances": MAX_CORRUPT_VERB_APPEARANCES,
                "max_clean_verb_per_domain": MAX_CLEAN_VERB_PER_DOMAIN,
            },
            "n_domains": len(domain_pairs),
            "n_pairs": sum(len(v) for v in domain_pairs.values()),
        },
        "domains": domain_pairs,
        "details": domain_details,
    }

    OUTPUT_STEP3.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STEP3, "w", encoding="utf-8") as f:
        json.dump(step3_artifact, f, indent=2)

    print(f"Candidate pairs before domain caps: {len(candidate_rows)}")
    print(f"Domains retained: {len(domain_pairs)}")
    print(f"Total pairs retained: {step3_artifact['meta']['n_pairs']}")
    print(f"Saved Step 3 artifact to: {OUTPUT_STEP3.resolve()}")

    top_domains = sorted(domain_pairs.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
    print("\nTop domains by retained pairs:")
    for domain, pairs in top_domains:
        print(f"  {domain:<22} {len(pairs)} pairs")
import json
import random
import uuid
from itertools import product
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

OUTPUT_RAW_PAIRS        = Path("../dataset/semantic_meaningful/raw_pairs.jsonl")
MAX_TOTAL_ROWS         = 400000
MAX_ROWS_PER_VERB_PAIR  = 5
MIN_ITEMS_PER_FRAME_KEY = 3
RANDOM_SEED             = 42

STEP1_PATH = Path("../dataset/semantic_meaningful/step1_verbnet_verb_pools.json")
STEP2_PATH = Path("../dataset/semantic_meaningful/step2_wordnet_noun_pools.json")

if __name__ == "__main__":
    step1 = json.load(open(STEP1_PATH, "r", encoding="utf-8"))
    step2 = json.load(open(STEP2_PATH, "r", encoding="utf-8"))

    MODEL_NAME = step1["meta"]["tokenizer_model"]
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Noun pools — 1-token only (all nouns are 1-token by Step 2 design)
    animate_1tok   = step2["nouns"]["animate_pool"]["by_token_length"]["1"]["words"]
    inanimate_1tok = step2["nouns"]["inanimate_pool"]["by_token_length"]["1"]["words"]

    assert len(animate_1tok)   >= MIN_ITEMS_PER_FRAME_KEY, "Animate noun pool too small."
    assert len(inanimate_1tok) >= MIN_ITEMS_PER_FRAME_KEY, "Inanimate noun pool too small."

    # Verb pools — all token lengths, enforce alignment per pair
    verb_buckets = step1["verbs"]["by_token_length"]

    # Group corrupt verbs by token length for fast lookup
    corrupt_by_tok: dict[int, list[str]] = {
        int(n): bucket["corrupt"]
        for n, bucket in verb_buckets.items()
        if bucket.get("corrupt")
    }

    # Build one frame per token-aligned (clean_verb, corrupt_verb) pair
    frames = [
        {
            "domain":           "unassigned",
            "clean_verb":       clean_v,
            "corrupt_verb":     corrupt_v,
            "patients":         animate_1tok,
            "animate_agents":   animate_1tok,
            "inanimate_agents": inanimate_1tok,
        }
        for n, bucket in verb_buckets.items()
        for clean_v in bucket.get("clean", [])
        for corrupt_v in corrupt_by_tok.get(int(n), [])
    ]

    assert len(frames) > 0, "No token-aligned verb pairs found."
    print(f"Token-aligned verb pairs (frames): {len(frames)}")

    rows = []
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    parity_failures = 0
    collision_skips = 0

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(frames)

    for fr in frames:
        if len(rows) >= MAX_TOTAL_ROWS:
            break

        clean_verb   = fr["clean_verb"]
        corrupt_verb = fr["corrupt_verb"]
        pair_key     = (clean_verb, corrupt_verb)

        remaining = min(
            MAX_ROWS_PER_VERB_PAIR - pair_counts[pair_key],
            MAX_TOTAL_ROWS - len(rows)
        )
        if remaining <= 0:
            continue

        patients         = fr["patients"]
        animate_agents   = fr["animate_agents"]
        inanimate_agents = fr["inanimate_agents"]

        # Sample indices directly — never materialize the full product
        seen = set()
        attempts = 0
        max_attempts = remaining * 10

        while pair_counts[pair_key] < MAX_ROWS_PER_VERB_PAIR and len(rows) < MAX_TOTAL_ROWS and attempts < max_attempts:
            p  = rng.choice(patients)
            aa = rng.choice(animate_agents)
            ia = rng.choice(inanimate_agents)
            attempts += 1

            if p == aa:
                continue
            if (p, aa, ia) in seen:
                continue
            seen.add((p, aa, ia))

            clean_prefix   = f"The {p} was {clean_verb} by the"
            corrupt_prefix = f"The {p} was {corrupt_verb} by the"

            if len(tokenizer.encode(clean_prefix)) != len(tokenizer.encode(corrupt_prefix)):
                parity_failures += 1
                continue

            rows.append({
                "clean":           clean_prefix,
                "corrupt":         corrupt_prefix,
                "patient":         p,
                "clean_verb":      clean_verb,
                "corrupt_verb":    corrupt_verb,
                "animate_agent":   aa,
                "inanimate_agent": ia,
                "domain":          fr["domain"],
                "uid":             uuid.uuid4().hex,
            })
            pair_counts[pair_key] += 1

    rng.shuffle(rows)

    OUTPUT_RAW_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RAW_PAIRS, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Animate 1-token nouns:          {len(animate_1tok)}")
    print(f"Inanimate 1-token nouns:        {len(inanimate_1tok)}")
    print(f"Rows generated:                 {len(rows)}")
    print(f"Patient==agent collisions (full space): {collision_skips}")
    print(f"Token parity failures:          {parity_failures}")
    print(f"Saved to:                       {OUTPUT_RAW_PAIRS.resolve()}")
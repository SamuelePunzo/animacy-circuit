import json
import random
import uuid
from pathlib import Path

OUTPUT_RAW_PAIRS        = Path("../dataset/semantic_meaningful/raw_pairs_semantic.jsonl")
MAX_TOTAL_ROWS         = 400000
MIN_ITEMS_PER_FRAME_KEY = 3
RANDOM_SEED             = 42

SEMANTIC_GROUPS_PATH    = Path("../dataset/semantic_meaningful/semantic_groups.json")

if __name__ == "__main__":
    semantic_groups = json.load(open(SEMANTIC_GROUPS_PATH, "r", encoding="utf-8"))

    required_keys = (
        "name",
        "patients",
        "clean_verbs",
        "corrupt_verbs",
    )
    frame_specs = semantic_groups.get("frames")
    assert isinstance(frame_specs, list) and len(frame_specs) > 0, "semantic_groups.json has no 'frames' list."

    semantic_frames_total = len(frame_specs)
    skipped_invalid_frames = 0

    # Cross-product pairing inside each semantic frame.
    frames = []
    for frame in frame_specs:
        if any(k not in frame for k in required_keys):
            skipped_invalid_frames += 1
            continue

        domain = frame["name"]
        patients = frame["patients"]
        clean_verbs = frame["clean_verbs"]
        corrupt_verbs = frame["corrupt_verbs"]

        if not isinstance(domain, str) or not domain.strip():
            skipped_invalid_frames += 1
            continue
        if not isinstance(patients, list) or len(patients) < MIN_ITEMS_PER_FRAME_KEY:
            skipped_invalid_frames += 1
            continue
        if not isinstance(clean_verbs, list) or len(clean_verbs) == 0:
            skipped_invalid_frames += 1
            continue
        if not isinstance(corrupt_verbs, list) or len(corrupt_verbs) == 0:
            skipped_invalid_frames += 1
            continue

        for clean_verb in clean_verbs:
            for corrupt_verb in corrupt_verbs:
                frames.append({
                    "domain": domain,
                    "clean_verb": clean_verb,
                    "corrupt_verb": corrupt_verb,
                    "patients": patients,
                })

    expanded_verb_pair_frames = len(frames)
    assert expanded_verb_pair_frames > 0, "No semantic-group verb-pair frames generated."

    rows = []

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(frames)

    for fr in frames:
        if len(rows) >= MAX_TOTAL_ROWS:
            break

        clean_verb = fr["clean_verb"]
        corrupt_verb = fr["corrupt_verb"]
        patients = list(dict.fromkeys(fr["patients"]))

        # Exhaustive generation: one row per patient for each verb pair.
        for p in patients:
            if len(rows) >= MAX_TOTAL_ROWS:
                break

            clean_prefix = f"The {p} was {clean_verb} by the"
            corrupt_prefix = f"The {p} was {corrupt_verb} by the"

            rows.append({
                "clean": clean_prefix,
                "corrupt": corrupt_prefix,
                "patient": p,
                "clean_verb": clean_verb,
                "corrupt_verb": corrupt_verb,
                "domain": fr["domain"],
                "uid": uuid.uuid4().hex,
            })

    rng.shuffle(rows)

    OUTPUT_RAW_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RAW_PAIRS, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Semantic frames loaded:         {semantic_frames_total}")
    print(f"Semantic frames skipped:        {skipped_invalid_frames}")
    print(f"Expanded verb-pair frames:      {expanded_verb_pair_frames}")
    print(f"Rows generated:                 {len(rows)}")
    print(f"Saved to:                       {OUTPUT_RAW_PAIRS.resolve()}")
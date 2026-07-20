from transformers import AutoTokenizer
import json
import nltk
from pathlib import Path
from nltk.corpus import wordnet as wn
import inflect

from datagen_utils import (
    all_hyponyms, group_by_tok, is_valid_patient, words_from_synsets, ARTIFACT_PHYSICAL_ANCHORS, 
    serialize_pool, is_valid_patient
)

OUTPUT_STEP2 = Path("../dataset/semantic_meaningful/step2_wordnet_noun_pools.json")
OUTPUT_CURATED = Path("../dataset/semantic_meaningful/curated_vocabulary.json")
MODEL_NAME = "gpt2"
OUTPUT_STEP1 = Path("../dataset/semantic_meaningful/step1_verbnet_verb_pools.json")

if __name__ == "__main__":

    assert OUTPUT_STEP1.exists(), f"Missing Step 1 artifact: {OUTPUT_STEP1.resolve()}"
    with open(OUTPUT_STEP1, "r", encoding="utf-8") as f:
        step1_artifact = json.load(f)

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    if "tokenizer" not in globals():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    infl = inflect.engine()
    print(f"Tokenizer model: {MODEL_NAME}")
    print(f"Loaded Step 1 artifact: {OUTPUT_STEP1.resolve()}")
    print(f"Step 2 output: {OUTPUT_STEP2.resolve()}")
    print(f"Curated vocab output: {OUTPUT_CURATED.resolve()}")


    #The inanimate pool is built from two sources unioned together: 
    # natural phenomena (earthquakes, floods, lightning) and physical artifacts (vehicles, tools, weapons). 
    # This is intentional becase we want both natural forces and man-made objects 
    # as main source of plausible inanimate agents
    person_synsets = all_hyponyms(wn.synset("person.n.01"))
    natural_synsets = all_hyponyms(wn.synset("natural_phenomenon.n.01"))
    artifact_synsets = set()
    for anchor_name in ARTIFACT_PHYSICAL_ANCHORS:
        artifact_synsets |= all_hyponyms(wn.synset(anchor_name))

    animate_all = words_from_synsets(person_synsets, infl=infl)
    animate_all = {w for w in animate_all if is_valid_patient(w)}
    
    inanimate_natural_all = words_from_synsets(natural_synsets, infl=infl)
    inanimate_artifact_all = words_from_synsets(artifact_synsets, infl=infl, concrete_artifact_filter=True)
    inanimate_all = inanimate_natural_all.union(inanimate_artifact_all)

    overlap_all = animate_all.intersection(inanimate_all)
    if overlap_all:
        inanimate_all = inanimate_all - overlap_all

    assert animate_all.isdisjoint(inanimate_all), "Animate/inanimate pools overlap after disjointness filtering."

    animate_by_tok = group_by_tok(sorted(animate_all), tokenizer=tokenizer)
    inanimate_by_tok = group_by_tok(sorted(inanimate_all), tokenizer=tokenizer)

    animate_1tok = sorted(animate_by_tok.get(1, []))
    inanimate_1tok = sorted(inanimate_by_tok.get(1, []))

    assert set(animate_1tok).isdisjoint(set(inanimate_1tok)), "Animate/inanimate 1-token noun pools overlap."
    assert len(animate_1tok) > 0, "No 1-token animate nouns produced."
    assert len(inanimate_1tok) > 0, "No 1-token inanimate nouns produced."

    print(f"Animate noun candidates (all token lengths): {len(animate_all)}")
    print(f"Inanimate natural candidates (all token lengths): {len(inanimate_natural_all)}")
    print(f"Inanimate artifact candidates (all token lengths): {len(inanimate_artifact_all)}")
    print(f"Inanimate noun candidates (union, post-disjoint): {len(inanimate_all)}")
    print(f"Full-set overlap removed from inanimate pool: {len(overlap_all)}")
    print(f"Animate 1-token nouns: {len(animate_1tok)}")
    print(f"Inanimate 1-token nouns: {len(inanimate_1tok)}")

    print("\nSample animate 1-token nouns:", animate_1tok[:20])
    print("\nSample inanimate 1-token nouns:", inanimate_1tok[:20])

    animate_pool = serialize_pool(
        animate_by_tok,
        "Animate noun pool from WordNet hyponyms of person.n.01 (singularized).",
    )
    inanimate_pool = serialize_pool(
        inanimate_by_tok,
        "Inanimate agent pool from natural_phenomenon.n.01 and physical artifact descendants (singularized).",
    )

    # Ensure the exported 1-token pools are exactly the disjoint versions used downstream.
    animate_pool["by_token_length"]["1"] = {"words": animate_1tok}
    inanimate_pool["by_token_length"]["1"] = {"words": inanimate_1tok}

    step2_artifact = {
        "meta": {
            "phase": "Step 2 only",
            "source": "WordNet via nltk.corpus.wordnet",
            "tokenizer_model": MODEL_NAME,
            "token_length_uses_leading_space": True,
            "animate_root": "person.n.01",
            "inanimate_roots": ["natural_phenomenon.n.01", "artifact physical anchors"],
            "artifact_anchors": ARTIFACT_PHYSICAL_ANCHORS,
            "inflect_singularization": True,
            "removed_overlap_from_inanimate_count": len(overlap_all),
        },
        "nouns": {
            "animate_pool": animate_pool,
            "inanimate_pool": inanimate_pool,
        },
    }

    OUTPUT_STEP2.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STEP2, "w", encoding="utf-8") as f:
        json.dump(step2_artifact, f, indent=2)

    curated_vocabulary = {
        "meta": {
            "phase": "Step 1 + Step 2",
            "description": "VerbNet verb pools plus WordNet noun pools for animacy s-selection dataset generation.",
            "tokenizer_model": MODEL_NAME,
            "token_length_uses_leading_space": True,
            "number": "singular - all nouns stored in singular form",
        },
        "verbs": step1_artifact["verbs"],
        "animate_pool": animate_pool,
        "inanimate_pool": inanimate_pool,
        "human_pool": animate_pool,
        "force_pool": inanimate_pool,
    }

    OUTPUT_CURATED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CURATED, "w", encoding="utf-8") as f:
        json.dump(curated_vocabulary, f, indent=2)

    print(f"Saved Step 2 artifact to: {OUTPUT_STEP2.resolve()}")
    print(f"Updated curated vocabulary: {OUTPUT_CURATED.resolve()}")
    print(f"Animate 1-token pool size: {len(animate_1tok)}")
    print(f"Inanimate 1-token pool size: {len(inanimate_1tok)}")
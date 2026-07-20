from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm  # Aggiunto l'import per la progress bar

PROMPT_VERSION = "prefix_plausibility_v1"

@dataclass(frozen=True)
class PromptSpec:
    yes_candidates: tuple[str, ...] = (" Yes", "Yes", "yes", " yes")
    no_candidates: tuple[str, ...] = (" No", "No", "no", " no")

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def default_paths() -> dict[str, Path]:
    base = repo_root() / "dataset" / "semantic_meaningful"
    return {
        "raw": base / "raw_pairs_semantic.jsonl",
        "cache": base / "phase6_prefix_score_cache_local.jsonl",
        "scored": base / "scored_semantic_pairs.jsonl",
    }

def build_prompt(prefix: str) -> str:
    safe_prefix = json.dumps(prefix, ensure_ascii=False)
    return (
        "Does the following incomplete sentence prefix make logical and semantic sense so far?\n"
        "Ignore the fact that it ends abruptly.\n"
        "Answer strictly with 'Yes' or 'No'.\n"
        f"Prefix: {safe_prefix}\n"
        "Answer:"
    )

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)

def load_cache(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    cache: dict[tuple[str, str, str], dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec.get("uid")
            model_name = rec.get("model")
            prompt_version = rec.get("prompt_version")
            if not uid or not model_name or not prompt_version:
                continue
            cache[(uid, model_name, prompt_version)] = rec
    return cache

def chunked(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def prepare_model(model_name: str, device: str, dtype: str):
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    selected_dtype = dtype_map[dtype]
    if device == "auto":
        device_map = "auto"
    elif device == "cpu":
        device_map = {"": "cpu"}
    else:
        device_map = {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=selected_dtype,
        device_map=device_map,
    )
    model.eval()
    return model

def _candidate_token_ids(tokenizer, candidate_text: str) -> list[int]:
    ids = tokenizer.encode(candidate_text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Candidate {candidate_text!r} tokenized to empty ids")
    return ids

def _prompt_id_lists(tokenizer, prompts: list[str]) -> list[list[int]]:
    encoded = tokenizer(prompts, add_special_tokens=False)
    return encoded["input_ids"]

def _score_candidate_for_prompt_ids(
    model,
    pad_token_id: int,
    prompt_id_lists: list[list[int]],
    candidate_ids: list[int],
) -> torch.Tensor:
    batch_size = len(prompt_id_lists)
    candidate_len = len(candidate_ids)
    merged: list[list[int]] = [p + candidate_ids for p in prompt_id_lists]
    prompt_lens = [len(p) for p in prompt_id_lists]
    merged_lens = [len(m) for m in merged]
    max_len = max(merged_lens)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, ids in enumerate(merged):
        seq_len = len(ids)
        input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    sum_logprobs = torch.zeros((batch_size,), dtype=torch.float32, device=model_device)
    batch_index = torch.arange(batch_size, device=model_device)
    for j in range(candidate_len):
        positions = torch.tensor([pl + j - 1 for pl in prompt_lens], device=model_device)
        token_id = candidate_ids[j]
        step_logits = logits[batch_index, positions, :]
        token_logits = step_logits[:, token_id].float()
        log_denom = torch.logsumexp(step_logits.float(), dim=-1)
        sum_logprobs += token_logits - log_denom
    return sum_logprobs.cpu()

def score_yes_no(
    model,
    tokenizer,
    prompts: list[str],
    prompt_spec: PromptSpec,
) -> list[dict]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must expose a pad token id")
    prompt_ids = _prompt_id_lists(tokenizer, prompts)
    if any(len(ids) == 0 for ids in prompt_ids):
        raise ValueError("At least one prompt tokenized to empty ids")
    yes_variant_scores = []
    no_variant_scores = []
    for text in prompt_spec.yes_candidates:
        ids = _candidate_token_ids(tokenizer, text)
        yes_variant_scores.append(
            _score_candidate_for_prompt_ids(model, pad_token_id, prompt_ids, ids)
        )
    for text in prompt_spec.no_candidates:
        ids = _candidate_token_ids(tokenizer, text)
        no_variant_scores.append(
            _score_candidate_for_prompt_ids(model, pad_token_id, prompt_ids, ids)
        )
    yes_scores = torch.stack(yes_variant_scores, dim=0).max(dim=0).values
    no_scores = torch.stack(no_variant_scores, dim=0).max(dim=0).values
    delta = yes_scores - no_scores
    p_yes = torch.sigmoid(delta)
    output: list[dict] = []
    for i in range(len(prompts)):
        output.append(
            {
                "yes_logprob": float(yes_scores[i].item()),
                "no_logprob": float(no_scores[i].item()),
                "delta": float(delta[i].item()),
                "p_yes": float(p_yes[i].item()),
            }
        )
    return output

def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description="Filter raw prefix pairs using local yes/no logit plausibility scoring."
    )
    parser.add_argument("--raw-path", type=Path, default=paths["raw"])
    parser.add_argument("--cache-path", type=Path, default=paths["cache"])
    parser.add_argument("--scored-path", type=Path, default=paths["scored"])
    parser.add_argument(
        "--model",
        type=str,
        # default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        default="meta-llama/Meta-Llama-3-70B-Instruct",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clean-min", type=float, default=0.70)
    parser.add_argument("--corrupt-min", type=float, default=0.70)
    parser.add_argument("--max-new-uids", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    prompt_spec = PromptSpec()
    assert args.batch_size > 0, "batch-size must be positive"
    assert 0.0 <= args.clean_min <= 1.0, "clean-min must be in [0, 1]"
    assert 0.0 <= args.corrupt_min <= 1.0, "corrupt-min must be in [0, 1]"
    if not args.raw_path.exists():
        raise FileNotFoundError(f"Missing raw pairs file: {args.raw_path}")
    raw_rows = load_jsonl(args.raw_path)
    by_uid = {row["uid"]: row for row in raw_rows if "uid" in row}
    print(f"Raw rows loaded: {len(raw_rows):,}")
    print(f"Unique UIDs: {len(by_uid):,}")
    print(f"Model: {args.model}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Cache path: {args.cache_path.resolve()}")
    cache = load_cache(args.cache_path)
    existing = {
        uid
        for uid in by_uid
        if (uid, args.model, PROMPT_VERSION) in cache
    }
    missing_uids = [
        uid
        for uid in by_uid
        if (uid, args.model, PROMPT_VERSION) not in cache
    ]
    if args.max_new_uids > 0:
        missing_uids = missing_uids[: args.max_new_uids]
    print(f"Cached rows for this model/prompt: {len(existing):,}")
    print(f"UIDs to score this run: {len(missing_uids):,}")
    
    if missing_uids:
        tokenizer = prepare_tokenizer(args.model)
        model = prepare_model(args.model, device=args.device, dtype=args.dtype)
        total_batches = (len(missing_uids) + args.batch_size - 1) // args.batch_size
        print(f"Total scoring batches: {total_batches:,}")
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Sostituito il vecchio ciclo for con tqdm
        for uid_batch in tqdm(chunked(missing_uids, args.batch_size), total=total_batches, desc="Valutazione (Batch)", unit="batch"):
            clean_prompts = [build_prompt(by_uid[uid]["clean"]) for uid in uid_batch]
            corrupt_prompts = [build_prompt(by_uid[uid]["corrupt"]) for uid in uid_batch]
            clean_scores = score_yes_no(model, tokenizer, clean_prompts, prompt_spec)
            corrupt_scores = score_yes_no(model, tokenizer, corrupt_prompts, prompt_spec)
            records = []
            for i, uid in enumerate(uid_batch):
                rec = {
                    "uid": uid,
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "yes_candidates": list(prompt_spec.yes_candidates),
                    "no_candidates": list(prompt_spec.no_candidates),
                    "clean_yes_logprob": clean_scores[i]["yes_logprob"],
                    "clean_no_logprob": clean_scores[i]["no_logprob"],
                    "clean_delta": clean_scores[i]["delta"],
                    "clean_p_yes": clean_scores[i]["p_yes"],
                    "corrupt_yes_logprob": corrupt_scores[i]["yes_logprob"],
                    "corrupt_no_logprob": corrupt_scores[i]["no_logprob"],
                    "corrupt_delta": corrupt_scores[i]["delta"],
                    "corrupt_p_yes": corrupt_scores[i]["p_yes"],
                    "scored_at_utc": now_iso,
                }
                cache[(uid, args.model, PROMPT_VERSION)] = rec
                records.append(rec)
            append_jsonl(args.cache_path, records)
            
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    scored_rows: list[tuple[dict, dict]] = []
    for row in raw_rows:
        uid = row.get("uid")
        if not uid:
            continue
        rec = cache.get((uid, args.model, PROMPT_VERSION))
        if rec is None:
            continue
        scored_rows.append((row, rec))
        
    scored_output_rows: list[dict] = []
    clean_low_count = 0
    corrupt_low_count = 0
    both_low_count = 0
    
    for row, rec in scored_rows:
        clean_p_yes = float(rec["clean_p_yes"])
        corrupt_p_yes = float(rec["corrupt_p_yes"])
        
        clean_ok = clean_p_yes >= args.clean_min
        corrupt_ok = corrupt_p_yes >= args.corrupt_min
        
        simplified_record = {
            "uid": row.get("uid"),
            "model": rec["model"],
            "yes_candidates": list(prompt_spec.yes_candidates),
            "no_candidates": list(prompt_spec.no_candidates),
            "clean": row.get("clean"),
            "corrupt": row.get("corrupt"),
            "clean_p_yes": clean_p_yes,
            "corrupt_p_yes": corrupt_p_yes
        }
        
        if not clean_ok:
            clean_low_count += 1
        if not corrupt_ok:
            corrupt_low_count += 1
        if (not clean_ok) and (not corrupt_ok):
            both_low_count += 1

        scored_output_rows.append(simplified_record)
        
    print(f"Scored rows available: {len(scored_rows):,} / {len(raw_rows):,}")
    print(f"Rows to write: {len(scored_output_rows):,}")
    print(f"Rows below clean_min ({args.clean_min:.2f}): {clean_low_count:,}")
    print(f"Rows below corrupt_min ({args.corrupt_min:.2f}): {corrupt_low_count:,}")
    print(f"Rows below both thresholds: {both_low_count:,}")
    
    if scored_rows:
        args.scored_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.scored_path, "w", encoding="utf-8") as f:
            for item in scored_output_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved scored rows: {len(scored_output_rows):,}")
        print(f"Saved consolidated scored pairs: {args.scored_path.resolve()}")
    else:
        print("No scored rows available for this model/prompt. Skipping scored output write.")
        
if __name__ == "__main__":
    main()

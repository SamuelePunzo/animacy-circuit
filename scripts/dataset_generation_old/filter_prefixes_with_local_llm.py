from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


PROMPT_VERSION = "prefix_plausibility_v1"


@dataclass(frozen=True)
class PromptSpec:
    yes_candidates: tuple[str, ...] = (" Yes", "Yes")
    no_candidates: tuple[str, ...] = (" No", "No")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths() -> dict[str, Path]:
    base = repo_root() / "dataset" / "semantic_meaningful"
    return {
        "raw": base / "raw_pairs.jsonl",
        "cache": base / "phase6_prefix_score_cache_local.jsonl",
        "filtered": base / "filtered_pairs_prefix_local.jsonl",
        "rejected": base / "rejected_pairs_prefix_local.jsonl",
        "summary": base / "prefix_filter_summary_local.json",
    }


def build_prompt(prefix: str) -> str:
    return (
        "Is the following sentence prefix semantically meaningful?\n"
        "Answer strictly with 'Yes' or 'No'.\n"
        f"Prefix: {json.dumps(prefix, ensure_ascii=False)}"
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


def prepare_llama_cpp_model(
    model_path: Path,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
):
    if Llama is None:
        raise ImportError(
            "llama-cpp-python is not installed. Install it to use --backend llama_cpp."
        )

    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        logits_all=True,
        verbose=False,
    )


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

    # Sum log-probabilities of candidate tokens under teacher forcing.
    # For token t at absolute position pos, the predictive distribution is logits[pos - 1].
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


def _extract_choice_and_logprobs(payload):
    if isinstance(payload, dict):
        choices = payload.get("choices", [])
        if not choices:
            raise ValueError("No choices returned by llama_cpp completion")
        choice = choices[0]
        logprobs = choice.get("logprobs")
        if logprobs is None:
            raise ValueError("No logprobs in llama_cpp completion output")
        token_logprobs = logprobs.get("token_logprobs")
        if token_logprobs is None:
            raise ValueError("token_logprobs missing in llama_cpp completion output")
        return choice, token_logprobs
    raise ValueError("Unexpected llama_cpp completion output type")


def _score_candidate_with_llama_cpp(llm, prompt: str, candidate: str) -> float:
    full = prompt + candidate
    completion = llm.create_completion(
        prompt=full,
        max_tokens=0,
        temperature=0,
        echo=True,
        logprobs=1,
    )
    _, token_logprobs = _extract_choice_and_logprobs(completion)

    full_ids = llm.tokenize(full.encode("utf-8"), add_bos=False, special=False)
    cand_ids = llm.tokenize(candidate.encode("utf-8"), add_bos=False, special=False)

    cand_len = len(cand_ids)
    if cand_len == 0:
        raise ValueError(f"Candidate {candidate!r} tokenized to empty ids in llama_cpp")

    if len(full_ids) >= cand_len and full_ids[-cand_len:] != cand_ids:
        prompt_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=False)
        cand_len = max(1, len(full_ids) - len(prompt_ids))

    finite_logprobs = [lp for lp in token_logprobs if isinstance(lp, (int, float))]
    if len(finite_logprobs) < cand_len:
        raise ValueError("Not enough token_logprobs to score candidate in llama_cpp")

    return float(sum(finite_logprobs[-cand_len:]))


def score_yes_no_llama_cpp(
    llm,
    prompts: list[str],
    prompt_spec: PromptSpec,
) -> list[dict]:
    output: list[dict] = []
    for prompt in prompts:
        yes_scores = [
            _score_candidate_with_llama_cpp(llm, prompt, c)
            for c in prompt_spec.yes_candidates
        ]
        no_scores = [
            _score_candidate_with_llama_cpp(llm, prompt, c)
            for c in prompt_spec.no_candidates
        ]

        yes = max(yes_scores)
        no = max(no_scores)
        delta = yes - no
        p_yes = float(torch.sigmoid(torch.tensor(delta)).item())

        output.append(
            {
                "yes_logprob": float(yes),
                "no_logprob": float(no),
                "delta": float(delta),
                "p_yes": p_yes,
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
    parser.add_argument("--filtered-path", type=Path, default=paths["filtered"])
    parser.add_argument("--rejected-path", type=Path, default=paths["rejected"])
    parser.add_argument("--summary-path", type=Path, default=paths["summary"])

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "transformers", "llama_cpp"],
        default="auto",
    )
    parser.add_argument("--gguf-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=10)

    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    sorted_vals = sorted(values)
    n = len(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    return {
        "mean": float(sum(values) / n),
        "median": float(median),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
    }


def main() -> None:
    args = parse_args()
    prompt_spec = PromptSpec()

    assert args.batch_size > 0, "batch-size must be positive"
    assert 0.0 <= args.clean_min <= 1.0, "clean-min must be in [0, 1]"
    assert 0.0 <= args.corrupt_min <= 1.0, "corrupt-min must be in [0, 1]"

    if not args.raw_path.exists():
        raise FileNotFoundError(f"Missing raw pairs file: {args.raw_path}")

    backend = args.backend
    if backend == "auto":
        backend = "llama_cpp" if args.gguf_path is not None else "transformers"

    if backend == "llama_cpp" and args.gguf_path is None:
        raise ValueError("--gguf-path is required when using --backend llama_cpp")

    if backend == "llama_cpp" and not args.gguf_path.exists():
        raise FileNotFoundError(f"Missing gguf model file: {args.gguf_path}")

    raw_rows = load_jsonl(args.raw_path)
    by_uid = {row["uid"]: row for row in raw_rows if "uid" in row}

    print(f"Raw rows loaded: {len(raw_rows):,}")
    print(f"Unique UIDs: {len(by_uid):,}")
    print(f"Backend: {backend}")
    if backend == "transformers":
        print(f"Model: {args.model}")
    else:
        print(f"GGUF model: {args.gguf_path}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Cache path: {args.cache_path.resolve()}")

    cache = load_cache(args.cache_path)
    existing = {
        uid
        for uid in by_uid
        if (uid, (args.model if backend == "transformers" else str(args.gguf_path)), PROMPT_VERSION) in cache
    }

    model_id = args.model if backend == "transformers" else str(args.gguf_path)

    missing_uids = [
        uid
        for uid in by_uid
        if (uid, model_id, PROMPT_VERSION) not in cache
    ]
    if args.max_new_uids > 0:
        missing_uids = missing_uids[: args.max_new_uids]

    print(f"Cached rows for this model/prompt: {len(existing):,}")
    print(f"UIDs to score this run: {len(missing_uids):,}")

    if missing_uids:
        tokenizer = None
        model = None
        llm = None

        if backend == "transformers":
            tokenizer = prepare_tokenizer(args.model)
            model = prepare_model(args.model, device=args.device, dtype=args.dtype)
        else:
            llm = prepare_llama_cpp_model(
                model_path=args.gguf_path,
                n_ctx=args.n_ctx,
                n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers,
            )

        total_batches = (len(missing_uids) + args.batch_size - 1) // args.batch_size
        print(f"Total scoring batches: {total_batches:,}")

        now_iso = datetime.now(timezone.utc).isoformat()
        written = 0

        for bi, uid_batch in enumerate(chunked(missing_uids, args.batch_size), start=1):
            clean_prompts = [build_prompt(by_uid[uid]["clean"]) for uid in uid_batch]
            corrupt_prompts = [build_prompt(by_uid[uid]["corrupt"]) for uid in uid_batch]

            if backend == "transformers":
                clean_scores = score_yes_no(model, tokenizer, clean_prompts, prompt_spec)
                corrupt_scores = score_yes_no(model, tokenizer, corrupt_prompts, prompt_spec)
            else:
                clean_scores = score_yes_no_llama_cpp(llm, clean_prompts, prompt_spec)
                corrupt_scores = score_yes_no_llama_cpp(llm, corrupt_prompts, prompt_spec)

            records = []
            for i, uid in enumerate(uid_batch):
                rec = {
                    "uid": uid,
                    "model": model_id,
                    "backend": backend,
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
                cache[(uid, model_id, PROMPT_VERSION)] = rec
                records.append(rec)

            written += append_jsonl(args.cache_path, records)
            if bi % args.progress_every == 0 or bi == total_batches:
                print(f"Scored batches: {bi:,}/{total_batches:,} | cache appends this run: {written:,}")

        del model
        del llm
        if backend == "transformers" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored_rows: list[tuple[dict, dict]] = []
    for row in raw_rows:
        uid = row.get("uid")
        if not uid:
            continue
        rec = cache.get((uid, model_id, PROMPT_VERSION))
        if rec is None:
            continue
        scored_rows.append((row, rec))

    filtered_rows: list[dict] = []
    rejected_rows: list[dict] = []
    rejection_counts = {"clean_low": 0, "corrupt_low": 0, "both_low": 0}

    clean_vals: list[float] = []
    corrupt_vals: list[float] = []

    for row, rec in scored_rows:
        clean_p_yes = float(rec["clean_p_yes"])
        corrupt_p_yes = float(rec["corrupt_p_yes"])
        clean_vals.append(clean_p_yes)
        corrupt_vals.append(corrupt_p_yes)

        clean_ok = clean_p_yes >= args.clean_min
        corrupt_ok = corrupt_p_yes >= args.corrupt_min

        enriched = {
            **row,
            "score_model": rec["model"],
            "prompt_version": rec["prompt_version"],
            "clean_yes_logprob": rec["clean_yes_logprob"],
            "clean_no_logprob": rec["clean_no_logprob"],
            "clean_delta": rec["clean_delta"],
            "clean_p_yes": rec["clean_p_yes"],
            "corrupt_yes_logprob": rec["corrupt_yes_logprob"],
            "corrupt_no_logprob": rec["corrupt_no_logprob"],
            "corrupt_delta": rec["corrupt_delta"],
            "corrupt_p_yes": rec["corrupt_p_yes"],
        }

        if clean_ok and corrupt_ok:
            filtered_rows.append(enriched)
            continue

        reasons = []
        if not clean_ok:
            reasons.append("clean_low")
        if not corrupt_ok:
            reasons.append("corrupt_low")
        if len(reasons) == 2:
            rejection_counts["both_low"] += 1
        elif reasons:
            rejection_counts[reasons[0]] += 1

        enriched["rejection_reasons"] = reasons
        rejected_rows.append(enriched)

    print(f"Scored rows available: {len(scored_rows):,} / {len(raw_rows):,}")
    print(f"Kept rows: {len(filtered_rows):,}")
    print(f"Rejected rows: {len(rejected_rows):,}")

    if scored_rows:
        for out_path, rows in [
            (args.filtered_path, filtered_rows),
            (args.rejected_path, rejected_rows),
        ]:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for item in rows:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "meta": {
                "phase": "Step 6 local prefix filter",
                "description": "Local LLM logit-based plausibility filtering on clean/corrupt sentence prefixes.",
                "model": args.model,
                "backend": backend,
                "prompt_version": PROMPT_VERSION,
                "yes_candidates": list(prompt_spec.yes_candidates),
                "no_candidates": list(prompt_spec.no_candidates),
                "clean_min": args.clean_min,
                "corrupt_min": args.corrupt_min,
                "batch_size": args.batch_size,
                "device": args.device,
                "dtype": args.dtype,
                "gguf_path": str(args.gguf_path) if args.gguf_path is not None else None,
                "n_ctx": args.n_ctx,
                "n_threads": args.n_threads,
                "n_gpu_layers": args.n_gpu_layers,
            },
            "counts": {
                "raw_rows": len(raw_rows),
                "scored_rows": len(scored_rows),
                "kept_rows": len(filtered_rows),
                "rejected_rows": len(rejected_rows),
                "keep_rate_over_scored": (len(filtered_rows) / len(scored_rows)) if scored_rows else None,
                "rejection_reasons": rejection_counts,
            },
            "stats": {
                "clean_p_yes": summarize(clean_vals),
                "corrupt_p_yes": summarize(corrupt_vals),
            },
            "paths": {
                "raw": str(args.raw_path),
                "cache": str(args.cache_path),
                "filtered": str(args.filtered_path),
                "rejected": str(args.rejected_path),
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved filtered pairs: {args.filtered_path.resolve()}")
        print(f"Saved rejected pairs: {args.rejected_path.resolve()}")
        print(f"Saved summary: {args.summary_path.resolve()}")
    else:
        print("No scored rows available for this model/prompt. Skipping output writes.")


if __name__ == "__main__":
    main()

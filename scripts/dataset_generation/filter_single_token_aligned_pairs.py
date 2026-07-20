import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer

def filter_single_tokens(
    input_path: str | Path, 
    output_path: str | Path, 
    model_name: str = "gpt2",
    filter_patient: bool = True
) -> None:
    
    # Initialize the Hugging Face tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    def is_single_token(word: str) -> bool:
        # Prepend a space to account for standard BPE/SentencePiece word boundaries
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        return len(tokens) == 1

    kept_rows = []
    dropped_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            row = json.loads(line)
            clean_text = str(row["clean"])
            corrupt_text = str(row["corrupt"])
            
            try:
                # "The [PATIENT] was [VERB] by the" -> PATIENT is index 1, VERB is index 3
                clean_patient = clean_text.split(" ")[1]
                corrupt_patient = corrupt_text.split(" ")[1]
                clean_verb = clean_text.split(" ")[3]
                corrupt_verb = corrupt_text.split(" ")[3]
            except IndexError:
                # Fallback for malformed rows
                dropped_count += 1
                continue
                
            # 1. Base requirement: Both verbs must be exactly 1 token
            keep = is_single_token(clean_verb) and is_single_token(corrupt_verb)

            # 2. Optional requirement: Both patients must be exactly 1 token
            if filter_patient and keep:
                keep = is_single_token(clean_patient) and is_single_token(corrupt_patient)

            if keep:
                kept_rows.append(row)
            else:
                dropped_count += 1
                
    # Write the filtered data to the new file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in kept_rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"✅ Processing complete.")
    print(f"Kept rows:    {len(kept_rows):,}")
    print(f"Dropped rows: {dropped_count:,}")

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    base = repo_root() / "dataset" / "semantic_meaningful"
    # Example usage:
    # Change "gpt2" to your target model (e.g., "meta-llama/Llama-2-7b-hf")
    parser = argparse.ArgumentParser(
        description="Filter minimal pairs for single-token verbs (and optionally patients)."
    )
    parser.add_argument("--input", type=Path, default=base / "accepted_filtered_pairs.jsonl", help="Path to input JSONL")
    parser.add_argument("--output", type=Path, default=base / "filtered_single_token_pairs.jsonl", help="Path to output JSONL")
    parser.add_argument("--model-name", type=str, default="gpt2", help="Tokenizer model (e.g., gpt2, meta-llama/Llama-2-7b-hf)")
    parser.add_argument(
        "--filter-patient", 
        default=True,
        help="If set, ALSO requires the PATIENT (index 1) to be a single token."
    )
    
    args = parser.parse_args()
    
    filter_single_tokens(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        filter_patient=args.filter_patient
    )
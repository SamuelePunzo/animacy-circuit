import json
import math
import argparse
from pathlib import Path
from llama_cpp import Llama
from tqdm import tqdm

# --- CONFIGURAZIONI DI DEFAULT ---
CLEAN_MIN_THRESHOLD = 0.70
CORRUPT_MIN_THRESHOLD = 0.70
YES_CANDIDATES = [" Yes", "Yes", "yes", " yes"]
NO_CANDIDATES = [" No", "No", "no", " no"]

def parse_args():
    parser = argparse.ArgumentParser(description="Filtra dataset JSONL usando un modello GGUF in locale su CPU.")
    parser.add_argument("--model", type=str, default="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", 
                        help="Nome o percorso del file del modello GGUF")
    parser.add_argument("--input", type=str, default="../../dataset/semantic_meaningful/raw_pairs.jsonl", 
                        help="File JSONL di input")
    parser.add_argument("--output", type=str, default="filtered_pairs_local.jsonl", 
                        help="File dove salvare le righe mantenute")
    parser.add_argument("--rejected", type=str, default="rejected_pairs_local.jsonl", 
                        help="File dove salvare le righe scartate")
    parser.add_argument("--limit", type=int, default=0, 
                        help="Numero massimo di righe da processare (0 = elabora tutto)")
    return parser.parse_args()

# def build_prompt(prefix: str) -> str:
#     return f"""<|start_header_id|>user<|end_header_id|>

# Is the following sentence prefix semantically meaningful?
# Answer strictly with 'Yes' or 'No'.
# Prefix: "{prefix}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# """

def build_prompt(prefix: str) -> str:
    safe_prefix = json.dumps(prefix, ensure_ascii=False)
    # Prompt ottimizzato per far ignorare la troncatura al modello
    return (
        "Does the following incomplete sentence prefix make logical and semantic sense so far?\n"
        "Ignore the fact that it ends abruptly.\n"
        "Answer strictly with 'Yes' or 'No'.\n"
        f"Prefix: {safe_prefix}\n"
        "Answer:"
    )

def sigmoid(x: float) -> float:
    if x >= 100: return 1.0
    if x <= -100: return 0.0
    return 1 / (1 + math.exp(-x))

def get_yes_no_probabilities(llm, prompt: str) -> dict:
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=1,
        logprobs=50, 
        temperature=0.0
    )
    
    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

    yes_scores = [top_logprobs.get(word, -100.0) for word in YES_CANDIDATES]
    no_scores = [top_logprobs.get(word, -100.0) for word in NO_CANDIDATES]
    
    best_yes = max(yes_scores)
    best_no = max(no_scores)
    
    delta = best_yes - best_no
    p_yes = sigmoid(delta)
    
    return {
        "yes_logprob": best_yes,
        "no_logprob": best_no,
        "delta": delta,
        "p_yes": p_yes
    }

def main():
    args = parse_args()

    if not Path(args.input).exists():
        print(f"Errore: Il file di input '{args.input}' non esiste.")
        return

    print(f"Caricamento del modello '{args.model}' in RAM...")
    try:
        llm = Llama(
            model_path=args.model,
            n_ctx=512,
            logits_all=True,
            verbose=False 
        )
    except Exception as e:
        print(f"Errore critico nel caricamento del modello: {e}")
        return

    print("Modello pronto!\n")

    # Lettura del file
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Applicazione del limite se specificato
    if args.limit > 0:
        lines = lines[:args.limit]
        print(f"ATTENZIONE: Limite impostato. Verranno processate solo le prime {args.limit} righe.\n")
    else:
        print(f"Trovate {len(lines)} righe. Verranno processate tutte.\n")

    filtered_rows = []
    rejected_rows = []

    # Loop con tqdm per la barra di avanzamento
    # Abbiamo rimosso i print interni per non sporcare la barra a schermo
    for line in tqdm(lines, desc="Valutazione", unit="riga"):
        if not line.strip(): continue
        row = json.loads(line)
        
        clean_text = row.get("clean", "")
        corrupt_text = row.get("corrupt", "")
        
        clean_scores = get_yes_no_probabilities(llm, build_prompt(clean_text))
        corrupt_scores = get_yes_no_probabilities(llm, build_prompt(corrupt_text))
        
        clean_ok = clean_scores["p_yes"] >= CLEAN_MIN_THRESHOLD
        corrupt_ok = corrupt_scores["p_yes"] >= CORRUPT_MIN_THRESHOLD
        
        row["clean_p_yes"] = clean_scores["p_yes"]
        row["corrupt_p_yes"] = corrupt_scores["p_yes"]
        
        if clean_ok and corrupt_ok:
            filtered_rows.append(row)
        else:
            rejected_rows.append(row)

    # Salvataggio
    with open(args.output, "w", encoding="utf-8") as f:
        for item in filtered_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(args.rejected, "w", encoding="utf-8") as f:
        for item in rejected_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n--- ELABORAZIONE COMPLETATA ---")
    print(f"Righe analizzate: {len(lines)}")
    print(f"Righe mantenute : {len(filtered_rows)} -> salvate in {args.output}")
    print(f"Righe scartate  : {len(rejected_rows)} -> salvate in {args.rejected}")

if __name__ == "__main__":
    main()
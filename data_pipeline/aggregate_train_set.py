import argparse
import json
import random

# Assemble the training set of the next classifier iteration (paper Sec. 4.3):
# D_train(t) = D_train(0) U D_gen(t) — the seed set merged with the current
# iteration's generator output, globally shuffled and re-indexed.
# --raw_data must be the generator's own output (JSONL rows with an
# `augmented_prompt` field); do NOT pass a preference-pairs file here.
#
# Example (final iteration):
# python aggregate_train_set.py \
#   --raw_data ./data/iter2_gen.jsonl \
#   --seed_data ./data/seed_train.jsonl \
#   --raw_source_tag gen_iter2 \
#   --output ./data/final_train.jsonl
parser = argparse.ArgumentParser(description="Merge the generator output D_gen(t) with the seed set D_train(0) into the classifier training set (paper Sec. 4.3)")
parser.add_argument("--raw_data", type=str, required=True, help="Augmented data produced by the generator in this iteration")
parser.add_argument("--seed_data", type=str, required=True, help="Seed training set JSONL")
parser.add_argument("--raw_source_tag", type=str, default="gen_iter2", help="Source tag recorded for --raw_data rows")
parser.add_argument("--output", type=str, required=True, help="Output path of the assembled training set")
_args = parser.parse_args()

raw_data_path = _args.raw_data
seed_data_path = _args.seed_data
output_path = _args.output

all_data = []

# 1. Read raw_data into a list
with open(raw_data_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if line:
            try:
                item = json.loads(line)
                # Unify the record format; tolerate a JSON-array line by iterating it
                for it in (item if isinstance(item, list) else [item]):
                    all_data.append({
                        "prompt": it["augmented_prompt"],
                        "category": it["category"],
                        "label": it["label"],
                        "source": _args.raw_source_tag
                    })
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[warn] Skipping unparseable line {line_num} of raw_data: {e}")

if not all_data:
    raise SystemExit("[error] No usable records in --raw_data; expected generator output with an `augmented_prompt` field.")

print(f"[*] Read {len(all_data)} records from raw_data.")

# Track the running total to report per-file counts
current_len = len(all_data)

# 2. Read seed_data
with open(seed_data_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if line:
            try:
                item = json.loads(line)
                for it in (item if isinstance(item, list) else [item]):
                    all_data.append({
                        "prompt": it.get("prompt"),
                        "category": it.get("category"),
                        "label": it.get("label"),
                        "source": it.get("source")
                    })
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[warn] Skipping unparseable line {line_num} of seed_data: {e}")

print(f"[*] Read {len(all_data) - current_len} records from seed_data.")

# 3. Global shuffle
print(f"[*] Shuffling {len(all_data)} records...")
random.seed(42)  # fixed seed for reproducibility
random.shuffle(all_data)

# 4. Re-index and write out
print(f"[*] Re-indexing and saving to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as writer:
    for idx, item in enumerate(all_data, start=1):
        # Keep `id` as the first key for readability
        new_item = {"id": idx}
        new_item.update(item)
        writer.write(json.dumps(new_item, ensure_ascii=False) + '\n')

print("Done!")

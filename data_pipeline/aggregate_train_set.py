import argparse
import json
import random

# Assemble the training set of the next classifier iteration (paper Sec. 4.3):
# current-iteration generated data + level-1 (hard) samples of the previous
# iteration + seed data, globally shuffled and re-indexed.
#
# Example:
# python aggregate_train_set.py \
#   --raw_data ./data/iter2_gen_mdpo.jsonl \
#   --keep_data ./data/iter1_level1.jsonl \
#   --seed_data ./data/seed_train.jsonl \
#   --output ./data/iter2_train.jsonl
parser = argparse.ArgumentParser(description="Merge generated, level-1 and seed data into the next classifier training set")
parser.add_argument("--raw_data", type=str, required=True, help="Augmented data produced by the generator in this iteration")
parser.add_argument("--keep_data", type=str, required=True, help="Level-1 (hard) samples kept from the previous iteration")
parser.add_argument("--seed_data", type=str, required=True, help="Seed training set JSONL")
parser.add_argument("--raw_source_tag", type=str, default="gen_iter2", help="Source tag recorded for --raw_data rows")
parser.add_argument("--keep_source_tag", type=str, default="gen_iter1", help="Source tag recorded for --keep_data rows")
parser.add_argument("--output", type=str, required=True, help="Output path of the assembled training set")
_args = parser.parse_args()

raw_data_path = _args.raw_data
keep_data_path = _args.keep_data
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

print(f"[*] Read {len(all_data)} records from raw_data.")

# Track the running total to report per-file counts
current_len = len(all_data)

# 2. Read keep_data
with open(keep_data_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if line:
            try:
                item = json.loads(line)
                for it in (item if isinstance(item, list) else [item]):
                    all_data.append({
                        "prompt": it["augmented_prompt"],
                        "category": it["category"],
                        "label": it["label"],
                        "source": _args.keep_source_tag
                    })
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[warn] Skipping unparseable line {line_num} of keep_data: {e}")

print(f"[*] Read {len(all_data) - current_len} records from keep_data.")
current_len = len(all_data)

# 3. Read seed_data
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

# 4. Global shuffle
print(f"[*] Shuffling {len(all_data)} records...")
random.seed(42)  # fixed seed for reproducibility
random.shuffle(all_data)

# 5. Re-index and write out
print(f"[*] Re-indexing and saving to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as writer:
    for idx, item in enumerate(all_data, start=1):
        # Keep `id` as the first key for readability
        new_item = {"id": idx}
        new_item.update(item)
        writer.write(json.dumps(new_item, ensure_ascii=False) + '\n')

print("Done!")

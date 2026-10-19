import argparse
import json
import random


# For each seed (id + original_prompt), keep exactly n of its augmented
# candidates, so downstream pair construction sees a balanced k=4 grouping.
def flush_group(group_key, records, n, sample_mode, rng, writer):
    if not group_key:
        return

    if sample_mode == "first":
        chosen = records[:n]
    else:
        if len(records) <= n:
            chosen = records[:]
        else:
            # Sample randomly but write back in the original order so the
            # record content and structure stay untouched
            idxs = sorted(rng.sample(range(len(records)), n))
            chosen = [records[i] for i in idxs]

    if not chosen:
        return

    for row in chosen:
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_augmented_prompts(input_file, output_file, n=4, sample_mode="first", seed=42):
    rng = random.Random(seed)
    current_key = None
    current_records = []
    total_groups = 0

    with open(input_file, "r", encoding="utf-8") as fin, open(
        output_file, "w", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip corrupt lines so huge files never abort the run
                continue

            key = (data.get("id"), data.get("original_prompt"))
            if key != current_key:
                flush_group(current_key, current_records, n, sample_mode, rng, fout)
                if current_key is not None:
                    total_groups += 1
                current_key = key
                current_records = [data]
            else:
                current_records.append(data)

        flush_group(current_key, current_records, n, sample_mode, rng, fout)
        if current_key is not None:
            total_groups += 1

    print(f"done: {total_groups} groups -> {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Keep n augmented_prompt candidates per (id, original_prompt) group"
    )
    parser.add_argument(
        "--input",
        default="./data/iter2_gen.jsonl",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        default="./data/iter2_gen_4aug.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help="Number of augmented_prompt candidates kept per group",
    )
    parser.add_argument(
        "--mode",
        choices=["first", "random"],
        default="random",
        help="first: keep the first n; random: sample n at random",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for random mode"
    )

    args = parser.parse_args()
    extract_augmented_prompts(
        input_file=args.input,
        output_file=args.output,
        n=args.num,
        sample_mode=args.mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

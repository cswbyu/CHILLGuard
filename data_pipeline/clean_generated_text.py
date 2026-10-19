import json
import re
import argparse

# Remove the "【新指令】：" ("new instruction:") prefix that the generator
# occasionally emits at the beginning of a rewritten sample.
# Default paths (prefer passing --input / --output explicitly)
file_path_in = "./data/generated_raw.jsonl"
file_path_out = "./data/generated_cleaned.jsonl"

def clean_value(value, pattern):
    """
    Recursively clean strings inside the JSON structure.
    """
    if isinstance(value, str):
        # Apply the substitution and return the cleaned string
        return pattern.sub('', value)
    elif isinstance(value, list):
        return [clean_value(item, pattern) for item in value]
    elif isinstance(value, dict):
        return {k: clean_value(v, pattern) for k, v in value.items()}
    else:
        # Other types (int, float, bool, None) pass through unchanged
        return value

def clean_jsonl(input_file, output_file):
    # Match a leading "【新指令】" prefix followed by a Chinese/English colon
    pattern = re.compile(r'^【新指令】[：:]')

    processed_count = 0
    modified_lines_count = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            if not line.strip():
                continue

            data = json.loads(line)
            processed_count += 1

            # Serialize the original record to detect whether cleaning changed it
            original_str = json.dumps(data, ensure_ascii=False)

            # Recursively clean every field
            cleaned_data = clean_value(data, pattern)

            if json.dumps(cleaned_data, ensure_ascii=False) != original_str:
                modified_lines_count += 1

            fout.write(json.dumps(cleaned_data, ensure_ascii=False) + '\n')

    print(f"Done!")
    print(f"total lines read: {processed_count}")
    print(f"lines modified: {modified_lines_count}")
    print(f"cleaned data saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip a given prefix from all string fields of a JSONL file")
    parser.add_argument("--input", type=str, default=file_path_in, help="Input JSONL file path")
    parser.add_argument("--output", type=str, default=file_path_out, help="Output JSONL file path")

    args = parser.parse_args()
    clean_jsonl(args.input, args.output)

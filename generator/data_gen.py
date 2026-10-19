import json
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm

# Shared PE rewriting templates (single source of truth in templates.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates import get_prompts

# python data_gen.py --gpus 0,7 --gpu_use 0.47 \
#     --model_path ./checkpoints/generator_iter1 \
#     --input_file ./data/seed_train.jsonl \
#     --output_file ./data/iter2_gen.jsonl

# Generator model path, seed data and output file are provided via CLI arguments.
MODEL_PATH = None
INPUT_FILE = None
OUTPUT_FILE = None

def worker_process(gpu_id: str, gpu_use: float, items_chunk: list, queue: mp.Queue, model_path: str):
    # 1. Set the visible CUDA device for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # 2. Import vllm/transformers inside the child process to avoid
    #    initializing a CUDA context in the parent before spawning
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_use,
        max_model_len=20480
    )
    # Sampling strategy aligned with the paper: k=4 candidates per seed prompt
    sampling_params = SamplingParams(temperature=0.7, n=4, max_tokens=2048)

    # Format prompts
    formatted_prompts = []
    for item in items_chunk:
        sys_p, usr_p = get_prompts(item["prompt"], is_safe=(item["label"] == "安全"))
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p}
        ]
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            kw["enable_thinking"] = False
            prompt = tokenizer.apply_chat_template(messages, **kw)
        except TypeError:  # templates without a thinking section ignore the flag
            del kw["enable_thinking"]
            prompt = tokenizer.apply_chat_template(messages, **kw)
        formatted_prompts.append(prompt)

    CHUNK_SIZE = 128
    for i in range(0, len(items_chunk), CHUNK_SIZE):
        chunk_prompts = formatted_prompts[i : i + CHUNK_SIZE]
        chunk_items = items_chunk[i : i + CHUNK_SIZE]

        # Disable vLLM's own progress bar in workers; progress is tracked by the parent
        outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=False)

        # Push all generated texts into the queue; the parent process persists them
        for k, output in enumerate(outputs):
            original_item = chunk_items[k]
            for j, generated_output in enumerate(output.outputs):
                new_item = {
                    "id": original_item["id"],
                    "original_prompt": original_item["prompt"],
                    "augmented_prompt": generated_output.text.strip(),
                    "category": original_item["category"],
                    "label": original_item["label"],
                    "source": original_item["source"],
                    "candidate_index": j + 1
                }
                queue.put(json.dumps(new_item, ensure_ascii=False) + "\n")

    # Done: send the termination sentinel
    queue.put(None)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU VLLM Data Generation")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--gpu_use", type=float, default=0.47, help="Per-GPU memory utilization; 0.47 is recommended to avoid OOM")
    parser.add_argument("--model_path", type=str, required=True, help="Generator checkpoint (previous MDPO/DPO output or the initial base model)")
    parser.add_argument("--input_file", type=str, required=True, help="Seed data JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for the augmented samples")

    args = parser.parse_args()

    global MODEL_PATH, INPUT_FILE, OUTPUT_FILE
    MODEL_PATH = args.model_path
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file

    # spawn is required: forking would copy an already-initialized CUDA context
    mp.set_start_method('spawn', force=True)

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    gpu_use = args.gpu_use
    num_gpus = len(gpu_list)
    print(f"--> Data-parallel inference on {num_gpus} GPU(s): {gpu_list}")

    # Resume support: count already-generated candidates per seed id so that
    # only fully augmented seeds (>= 4 candidates, matching n=4) are skipped
    processed_counts = {}
    if os.path.exists(OUTPUT_FILE):
        print(f"--> Found existing output file {OUTPUT_FILE}, reading progress...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "id" in data:
                            pid = data["id"]
                            processed_counts[pid] = processed_counts.get(pid, 0) + 1
                    except json.JSONDecodeError:
                        continue

    processed_ids = {pid for pid, count in processed_counts.items() if count >= 4}
    print(f"--> {len(processed_ids)} seed prompts already fully augmented (4 candidates each).")

    seed_dataset = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["id"] not in processed_ids:
                    seed_dataset.append(item)

    if len(seed_dataset) == 0:
        print("All seed prompts are already augmented; nothing to do.")
        return

    print(f"--> {len(seed_dataset)} seed prompts remaining; splitting evenly across GPUs...")

    # Split the workload across GPUs
    chunk_size = (len(seed_dataset) + num_gpus - 1) // num_gpus
    worker_chunks = [seed_dataset[i : i + chunk_size] for i in range(0, len(seed_dataset), chunk_size)]

    # Inter-process queue for the generated texts
    queue = mp.Queue(maxsize=10000)
    processes = []

    for i, chunk in enumerate(worker_chunks):
        if i >= num_gpus: break
        p = mp.Process(target=worker_process, args=(gpu_list[i], gpu_use, chunk, queue, MODEL_PATH))
        p.start()
        processes.append(p)

    # The parent process drains the queue and appends results to the output file
    finished_workers = 0
    augmented_count = 0
    total_expected_augmented = len(seed_dataset) * 4  # n=4 candidates per seed

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f, tqdm(total=total_expected_augmented, desc="augmentation progress") as pbar:
        while finished_workers < len(processes):
            item_str = queue.get()
            if item_str is None:
                finished_workers += 1
            else:
                f.write(item_str)
                f.flush()  # flush eagerly so an interruption loses no data
                augmented_count += 1
                pbar.update(1)

    for p in processes:
        p.join()

    print(f"\nAll workers finished. Generated {augmented_count} samples, appended to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()

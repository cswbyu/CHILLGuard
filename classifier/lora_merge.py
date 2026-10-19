import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Physically merge a LoRA adapter into its base model")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base model path, e.g. ./models/Qwen3-8B")
    parser.add_argument("--lora_model_path", type=str, required=True,
                        help="LoRA adapter weights (SFT / MDPO training output)")
    parser.add_argument("--output_merged_path", type=str, required=True,
                        help="Output directory for the merged full model")
    parser.add_argument("--max_memory_per_gpu", type=str, default="48GB",
                        help="Per-GPU memory cap; adjust to your hardware, pass 'none' for no limit")
    return parser.parse_args()


# Paths are injected from the command line in main()
base_model_path = None
lora_model_path = None
output_merged_path = None

# ==========================================
# Multi-GPU configuration
# ==========================================
# Recommended: set CUDA_VISIBLE_DEVICES before launching, e.g.
# CUDA_VISIBLE_DEVICES=5 python lora_merge.py --base_model_path ... --lora_model_path ... --output_merged_path ...
# so the script sees logical GPUs 0 and 1.
use_multi_gpu_auto = True
max_memory_per_gpu = "48GB"  # adjust to your VRAM; set to None for no limit


def build_max_memory_map(limit_per_gpu: str | None):
    if not limit_per_gpu:
        return None
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return None
    return {i: limit_per_gpu for i in range(gpu_count)}

def main():
    global base_model_path, lora_model_path, output_merged_path, max_memory_per_gpu
    args = parse_args()
    base_model_path = args.base_model_path
    lora_model_path = args.lora_model_path
    output_merged_path = args.output_merged_path
    max_memory_per_gpu = None if args.max_memory_per_gpu.lower() == "none" else args.max_memory_per_gpu

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device detected; run this script on a GPU machine.")

    visible_gpus = torch.cuda.device_count()
    print(f"[*] Visible GPUs: {visible_gpus}")
    print(f"[*] 1. Loading base model: {base_model_path}")

    load_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if use_multi_gpu_auto:
        load_kwargs["device_map"] = "auto"
        max_memory_map = build_max_memory_map(max_memory_per_gpu)
        if max_memory_map is not None:
            load_kwargs["max_memory"] = max_memory_map

    # device_map="auto" shards the base model across the visible GPUs
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **load_kwargs,
    )

    print(f"[*] 2. Attaching LoRA weights: {lora_model_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        dtype=torch.bfloat16,
    )

    print("[*] 3. Merging on GPU (merge_and_unload)...")
    # The matrix merge (base weights + scaled LoRA weights) runs on CUDA cores
    merged_model = peft_model.merge_and_unload()

    print(f"[*] 4. Saving the merged full model to: {output_merged_path}")
    merged_model.save_pretrained(output_merged_path, safe_serialization=True)

    print("[*] 5. Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_merged_path)

    print("\n[+] Done! GPU merge complete.")

if __name__ == "__main__":
    main()

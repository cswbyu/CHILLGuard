import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

# Single-GPU example:
# accelerate launch --num_processes=1 --gpu_ids 6 lora_sft_trainer.py --config ./configs/sft_config.json 2>&1 | tee lora_single.log
# Multi-GPU data-parallel example:
# accelerate launch --num_processes=2 --gpu_ids 2,5 lora_sft_trainer.py --config ./configs/sft_config.json --multi_gpu 2>&1 | tee lora_multi.log
#
# LoRA variant of the classifier SFT (memory-efficient alternative to sft_trainer.py).
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT Trainer (supports multi-GPU data parallel)")
    parser.add_argument("--config", type=str, default="./configs/sft_config.json", help="Path to the training config file")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU data parallelism (DDP); launch via accelerate/torchrun")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def build_sft_prompt(tokenizer, user_text, assistant_text):
    messages = [
        {"role": "system", "content": "你是一个安全审核助手。请只输出'安全'或'不安全'。"},
        {"role": "user", "content": f"请判断以下文本是否安全：\n{user_text}"},
        {"role": "assistant", "content": assistant_text}
    ]
    # NOTE: the chat template above is part of the method and is intentionally
    # kept in Chinese. No add_generation_prompt=True for full/LoRA SFT.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False
    )

def prepare_dataset(train_file, eval_file, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_raw = load_dataset("json", data_files=train_file, split="train")
    eval_raw = load_dataset("json", data_files=eval_file, split="train")

    def map_sft_format(example):
        full_text = build_sft_prompt(tokenizer, example["prompt"], example["label"])
        return {"text": full_text}

    train_dataset = train_raw.map(map_sft_format, remove_columns=train_raw.column_names)
    eval_dataset = eval_raw.map(map_sft_format, remove_columns=eval_raw.column_names)

    return train_dataset, eval_dataset

def main():
    cli_args = parse_args()
    config = load_config(cli_args.config)
    torch.set_float32_matmul_precision('high')

    enable_multi_gpu = cli_args.multi_gpu or config.get("multi_gpu", False)
    world_size = get_world_size()
    rank = get_rank()

    if enable_multi_gpu and world_size == 1:
        print("[!] Warning: multi_gpu is enabled in the config/CLI but WORLD_SIZE=1; falling back to single-GPU training.")
        enable_multi_gpu = False

    if rank == 0:
        mode = "DDP multi-GPU" if enable_multi_gpu else "single-GPU"
        print(f"[*] Training mode: {mode} | world_size={world_size}")

    model_id = config["model_id"]
    dtype = getattr(torch, config["dtype"])

    # 1. Load the base model
    # Note: do NOT call .gradient_checkpointing_enable() manually here
    print("[*] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    # Disable the KV cache explicitly; prerequisite for gradient checkpointing
    model.config.use_cache = False

    # 2. LoRA configuration
    # The model is not wrapped with get_peft_model here; the config is handed
    # to the trainer, which injects the adapters itself.
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),               # r=16 recommended
        lora_alpha=config.get("lora_alpha", 32),  # usually 2x r
        target_modules="all-linear",              # all linear layers works best
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Data
    train_file = config.get("train_file")
    eval_file = config.get("eval_file")
    train_dataset, eval_dataset = prepare_dataset(train_file, eval_file, model_id)

    if rank == 0:
        print(f"[*] SFT dataset ready:")
        print(f"    - train samples: {len(train_dataset)}")
        print(f"    - eval samples: {len(eval_dataset)}")

    # 5. SFT training arguments
    sft_kwargs = dict(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        optim=config["optim"],
        bf16=config["bf16"],
        fp16=config["fp16"],
        tf32=config["tf32"],
        eval_strategy="steps",
        eval_steps=config.get("save_steps", 500),
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        dataset_text_field="text",
        max_length=config.get("max_length", 4096),
        report_to=config["report_to"],
        gradient_checkpointing=True,        # SFTTrainer enables it safely through this flag
        packing=False,
    )

    # DDP-related options; single-GPU runs simply ignore them
    if enable_multi_gpu:
        sft_kwargs["ddp_find_unused_parameters"] = config.get("ddp_find_unused_parameters", False)
        ddp_backend = config.get("ddp_backend", "nccl")
        if ddp_backend:
            sft_kwargs["ddp_backend"] = ddp_backend

    sft_args = SFTConfig(**sft_kwargs)

    # 6. Trainer
    if rank == 0:
        print("[*] Initializing trainer (LoRA adapters injected automatically)...")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config             # <-- key point: pass lora_config to the trainer
    )

    # Print the trainable parameter count to confirm LoRA injection
    if rank == 0:
        trainer.model.print_trainable_parameters()

    # 7. Train
    if rank == 0:
        print("[*] Starting SFT training with evaluation...")
    trainer.train()

    # 8. Save
    # With LoRA only the adapter weights are saved, not the full base model
    final_save_path = os.path.join(config["output_dir"], config["output_model_name"])
    trainer.save_model(final_save_path)
    if rank == 0:
        print(f"[*] SFT training finished. LoRA weights saved to: {final_save_path}")

if __name__ == "__main__":
    main()

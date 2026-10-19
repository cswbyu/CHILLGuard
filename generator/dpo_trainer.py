import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOTrainer, DPOConfig
from accelerate.state import PartialState

# Shared PE rewriting templates (single source of truth in templates.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates import get_prompts

# Launch commands:
# accelerate launch --num_processes=1 --gpu_ids 0 dpo_trainer.py 2>&1 | tee dpo_base.log
# CUDA_VISIBLE_DEVICES=0 python dpo_trainer.py 2>&1 | tee output.log
#
# Standard DPO baseline for the generator (ablation variant CHILLGuard‡
# in Table 3 of the paper, i.e. MDPO replaced by static-beta DPO).

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if isinstance(config_data, dict) and "dpo_base" in config_data:
        return config_data["dpo_base"]

    return config_data

def build_dpo_prompt(tokenizer, seed_text: str, is_safe: bool):
    """
    Build the prompt part (system + user).
    The assistant responses (chosen / rejected) are appended by DPOTrainer.
    """
    system_prompt, user_prompt = get_prompts(seed_text, is_safe)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def prepare_dataset(train_file, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_raw = load_dataset("json", data_files=train_file, split="train")

    def map_dpo_format(example):
        original_label = example.get("original_label", "").strip()
        is_safe = (original_label == "安全")
        seed_text = example["prompt"]

        return {
            "prompt": build_dpo_prompt(tokenizer, seed_text, is_safe),
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    train_dataset = train_raw.map(map_dpo_format, remove_columns=train_raw.column_names)
    train_dataset = train_dataset.filter(lambda x: x["chosen"] is not None and x["rejected"] is not None)

    return train_dataset


def main():
    config = load_config("./configs/dpo_config.json")
    torch.set_float32_matmul_precision('high')

    model_id = config["model_id"]
    dtype = getattr(torch, config["dtype"])

    print(f"[*] Loading base model for DPO: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": PartialState().process_index},
        trust_remote_code=True
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_file = config.get("train_file")

    train_dataset = prepare_dataset(train_file, model_id)

    print(f"[*] Dataset ready:")
    print(f"    - train samples: {len(train_dataset)}")

    dpo_args = DPOConfig(
        output_dir=config["output_dir"],
        beta=config.get("beta", 0.1),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config.get("learning_rate", 5e-6),
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        optim=config["optim"],
        bf16=config["bf16"],
        fp16=config["fp16"],
        tf32=config.get("tf32", False),
        eval_strategy="no",
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        report_to=config.get("report_to", "none"),
        gradient_checkpointing=True,
        max_prompt_length=config.get("max_prompt_length", 512),
        max_length=config.get("max_length", 1024),
        remove_unused_columns=False,
    )

    print("[*] Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("[*] Starting DPO training...")
    trainer.train()

    final_save_path = os.path.join(config["output_dir"], config["output_model_name"])
    trainer.save_model(final_save_path)
    print(f"[*] DPO training finished. Model saved to: {final_save_path}")

if __name__ == "__main__":
    main()

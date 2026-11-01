import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType

# Shared PE rewriting templates (single source of truth in templates.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates import get_prompts

# Launch commands:
# accelerate launch --num_processes=1 --gpu_ids 0 dpo_lora_trainer.py 2>&1 | tee dpo_lora.log
# CUDA_VISIBLE_DEVICES=1 python dpo_lora_trainer.py 2>&1 | tee output.log
#
# LoRA variant of the standard DPO baseline (ablation, Table 3 of the paper).

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if isinstance(config_data, dict) and "dpo_lora" in config_data:
        return config_data["dpo_lora"]

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
    # add_generation_prompt=True so the model is ready to emit the assistant turn
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def prepare_dataset(train_file, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # The dataset is expected in JSON Lines (.jsonl) or an equivalent .json format
    train_raw = load_dataset("json", data_files=train_file, split="train")

    def map_dpo_format(example):
        # Safety side of the pair is determined by the original seed label
        original_label = example.get("original_label", "").strip()
        is_safe = (original_label == "安全")

        # The seed text lives in the "seed_prompt" field of the pair file
        seed_text = example["seed_prompt"]

        return {
            "prompt": build_dpo_prompt(tokenizer, seed_text, is_safe),
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    # Map down to the prompt / chosen / rejected columns expected by DPOTrainer
    train_dataset = train_raw.map(map_dpo_format, remove_columns=train_raw.column_names)

    # Drop entries with missing responses
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
        trust_remote_code=True
    )
    # Disable the KV cache explicitly; required by gradient checkpointing
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules="all-linear",
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_file = config.get("train_file")

    # Training split only
    train_dataset = prepare_dataset(train_file, model_id)

    print(f"[*] Dataset ready:")
    print(f"    - train samples: {len(train_dataset)}")

    dpo_args = DPOConfig(
        output_dir=config["output_dir"],
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
        eval_strategy="no",  # no validation during DPO
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        report_to=config.get("report_to", "none"),
        gradient_checkpointing=True,
        beta=config.get("beta", config.get("dpo_beta", 0.1)),
        max_length=config.get("max_length", 1024),
        remove_unused_columns=False
    )

    print("[*] Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config
    )

    trainer.model.print_trainable_parameters()

    print("[*] Starting DPO training...")
    trainer.train()

    final_save_path = os.path.join(config["output_dir"], config["output_model_name"])
    trainer.save_model(final_save_path)
    print(f"[*] DPO training finished. LoRA weights saved to: {final_save_path}")

if __name__ == "__main__":
    main()

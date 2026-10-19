import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
# accelerate launch --num_processes=2 --gpu_ids 5,6 sft_trainer.py 2>&1 | tee output.log
# CUDA_VISIBLE_DEVICES=1 python sft_trainer.py
#
# Full-parameter SFT of the guardrail classifier (paper Sec. 4.1 / 4.3):
# the classifier is fine-tuned from scratch in every iteration on the
# training set assembled for that iteration.


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def build_sft_prompt(tokenizer, user_text, assistant_text):

    messages = [
        {"role": "system", "content": "你是一个安全审核助手。请只输出'安全'或'不安全'。"},
        {"role": "user", "content": f"请判断以下文本是否安全：\n{user_text}"},
        {"role": "assistant", "content": assistant_text}
    ]
    # NOTE: the chat template above is part of the method and is intentionally
    # kept in Chinese. No add_generation_prompt=True here: SFT is full-sequence
    # training and must include the assistant response.
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
    # Load config
    config = load_config("./configs/sft_config.json")
    torch.set_float32_matmul_precision('high')

    model_id = config["model_id"]
    dtype = getattr(torch, config["dtype"])

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-side padding is recommended for SFT
    tokenizer.padding_side = "right"

    train_file = config.get("train_file")
    eval_file = config.get("eval_file")

    train_dataset, eval_dataset = prepare_dataset(train_file, eval_file, model_id)

    print(f"[*] SFT dataset ready:")
    print(f"    - train samples: {len(train_dataset)}")
    print(f"    - eval samples: {len(eval_dataset)}")

    # SFT training arguments
    sft_args = SFTConfig(
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
        max_length=config.get("max_length", 4096),
        save_steps=config["save_steps"],
        dataset_text_field="text",  # name of the text column in the dataset
        report_to=config["report_to"],
        gradient_checkpointing=True,
        packing=False  # can be set to True to speed up training on short texts
    )

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
        )

    print("[*] Starting SFT training with evaluation...")
    trainer.train()

    final_save_path = os.path.join(config["output_dir"], config["output_model_name"])
    trainer.save_model(final_save_path)
    print(f"SFT training finished. Model saved to: {final_save_path}")

if __name__ == "__main__":
    main()

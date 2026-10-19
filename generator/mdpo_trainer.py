import os
import json
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Dict, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType
from accelerate.state import PartialState

# Shared PE rewriting templates (single source of truth in templates.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates import get_prompts
# CUDA_VISIBLE_DEVICES=7 python mdpo_trainer.py
# Custom MDPOTrainer: DPO with the model-aware dynamic KL penalty (MDPO, paper Sec. 4.2).
class MDPOTrainer(DPOTrainer):
    def __init__(self, *args, gamma=0.9, K=6, **kwargs):
        # Extract MDPO hyper-parameters from the training config
        super().__init__(*args, **kwargs)
        self.beta_base = self.args.beta
        self.gamma = gamma
        self.K = K
        self.running_reward_gap = torch.tensor(
            1.0, device=self.args.device
        )
        self.current_dynamic_beta = self.beta_base

    def get_batch_logps(self, logits, labels, average_log_prob=False):
        """Compute log probabilities for the given labels."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2).clamp(min=0)).squeeze(2)
        mask = (shift_labels != -100)
        per_token_logps = per_token_logps * mask
        if average_log_prob:
            return per_token_logps.sum(-1) / mask.sum(-1)
        return per_token_logps.sum(-1)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Core MDPO loss (DPO + model-aware dynamic KL weighting).
        """

        # ===== Build labels: mask out padding tokens =====
        def build_labels(input_ids, attention_mask):
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            return labels

        # ===== Padding helper: align chosen / rejected seq_len =====
        def pad_to_max_len(input_ids, attention_mask, max_len, pad_token_id):
            pad_len = max_len - input_ids.size(1)
            if pad_len > 0:
                input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            return input_ids, attention_mask

        # ===== batch size =====
        batch_size = inputs["chosen_input_ids"].size(0)

        # Raw inputs
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_mask = inputs["rejected_attention_mask"]

        # ===== 1. Align sequence lengths =====
        max_len = max(chosen_ids.size(1), rejected_ids.size(1))
        pad_id = self.processing_class.pad_token_id

        chosen_ids, chosen_mask = pad_to_max_len(chosen_ids, chosen_mask, max_len, pad_id)
        rejected_ids, rejected_mask = pad_to_max_len(rejected_ids, rejected_mask, max_len, pad_id)

        # ===== 2. Build labels =====
        chosen_labels = build_labels(chosen_ids, chosen_mask)
        rejected_labels = build_labels(rejected_ids, rejected_mask)

        # Concatenate chosen + rejected into a single forward pass for efficiency
        all_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        all_attention_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
        all_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        # ===== 3. Policy model log-probs =====
        logits = model(all_ids, attention_mask=all_attention_mask).logits
        all_logps = self.get_batch_logps(logits, all_labels)

        policy_chosen_logps = all_logps[:batch_size]
        policy_rejected_logps = all_logps[batch_size:]

        # ===== 4. Reference model log-probs (no grad) =====
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logits = self.ref_model(all_ids, attention_mask=all_attention_mask).logits
            else:
                # Under LoRA training TRL may not build an explicit ref_model;
                # obtain the reference distribution with the adapter disabled.
                if hasattr(self, "null_ref_context"):
                    ref_ctx = self.null_ref_context()
                elif hasattr(model, "disable_adapter"):
                    ref_ctx = model.disable_adapter()
                else:
                    ref_ctx = nullcontext()

                with ref_ctx:
                    ref_logits = model(all_ids, attention_mask=all_attention_mask).logits

            ref_logps = self.get_batch_logps(ref_logits, all_labels)
            ref_chosen_logps = ref_logps[:batch_size]
            ref_rejected_logps = ref_logps[batch_size:]

        # ===== 5. Reward gap (preference strength), Eq. 3 =====
        reward_gaps = self.beta_base * (
            (policy_chosen_logps - ref_chosen_logps)
            - (policy_rejected_logps - ref_rejected_logps)
        )

        # ===== 6. Outlier filtering (Eq. 4-5): keep the K samples closest to the batch center =====
        normalized_gaps = reward_gaps / (self.running_reward_gap + 1e-8)

        center = normalized_gaps.mean().detach()
        dist_sq = (normalized_gaps - center) ** 2
        _, stable_idx = torch.topk(
            dist_sq, k=min(self.K, batch_size), largest=False
        )
        batch_avg_gap = normalized_gaps[stable_idx].mean()

        # ===== 7. Model-aware dynamic beta (Eq. 6) =====
        alpha_m = torch.sigmoid(batch_avg_gap) / torch.sigmoid(
            self.running_reward_gap.detach()
        )
        dynamic_beta = self.beta_base * alpha_m

        self.current_dynamic_beta = dynamic_beta.item()

        # ===== 8. DPO loss with the dynamic KL coefficient =====
        logits_diff = (
            (policy_chosen_logps - ref_chosen_logps)
            - (policy_rejected_logps - ref_rejected_logps)
        )
        losses = -F.logsigmoid(dynamic_beta * logits_diff)

        # ===== 9. Momentum update of the global reward gap (Eq. 7) =====
        if model.training:
            self.running_reward_gap = (
                self.gamma * self.running_reward_gap
                + (1 - self.gamma) * batch_avg_gap.detach()
            )

        return losses.mean()

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override `log` to match the signature of newer transformers versions.
        """
        if logs is not None:
            logs["dynamic_beta"] = self.current_dynamic_beta
            logs["running_reward_gap"] = self.running_reward_gap.item()

        # Forward all arguments to the base class for compatibility
        super().log(logs, start_time)

    def _save_checkpoint(self, model, trial):
        """Record the momentum state when saving a checkpoint."""
        self.state.combined_reward_gap = self.running_reward_gap.item()
        super()._save_checkpoint(model, trial)

    def _load_from_checkpoint(self, resume_from_checkpoint):
        """Restore the momentum state when resuming from a checkpoint."""
        super()._load_from_checkpoint(resume_from_checkpoint)

        if hasattr(self.state, "combined_reward_gap"):
            self.running_reward_gap = torch.tensor(
                self.state.combined_reward_gap,
                device=self.args.device
            )

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if isinstance(config_data, dict):
        if "mdpo_lora" in config_data:
            return config_data["mdpo_lora"]
        if "mdpo_base" in config_data:
            return config_data["mdpo_base"]
        if "dpo_base" in config_data:
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

def prepare_dataset(train_file, model_id, eval_file=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_raw = load_dataset("json", data_files=train_file, split="train")

    def map_dpo_format(example):
        original_label = example.get("original_label", "").strip()
        is_safe = (original_label == "安全")
        seed_text = example["seed_prompt"]

        return {
            "prompt": build_dpo_prompt(tokenizer, seed_text, is_safe),
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    train_dataset = train_raw.map(map_dpo_format, remove_columns=train_raw.column_names)
    train_dataset = train_dataset.filter(lambda x: x["chosen"] is not None and x["rejected"] is not None)

    eval_dataset = None
    if eval_file:
        eval_raw = load_dataset("json", data_files=eval_file, split="train")
        eval_dataset = eval_raw.map(map_dpo_format, remove_columns=eval_raw.column_names)
        eval_dataset = eval_dataset.filter(lambda x: x["chosen"] is not None and x["rejected"] is not None)

    return train_dataset, eval_dataset

def main():
    # Load config
    config = load_config("./configs/mdpo_config.json")
    torch.set_float32_matmul_precision('high')

    model_id = config["model_id"]
    dtype = getattr(torch, config["dtype"])

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

    use_lora = config.get("use_lora", True)
    lora_config = None
    if use_lora:
        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", "all-linear"),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias=config.get("lora_bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

    train_file = config.get("train_file")
    eval_strategy = config.get("eval_strategy", "no")
    eval_file = config.get("eval_file") if eval_strategy != "no" else None

    train_dataset, eval_dataset = prepare_dataset(train_file, model_id, eval_file)

    print(f"[*] Dataset ready:")
    print(f"    - train samples: {len(train_dataset)}")

    if eval_dataset is not None:
        print(f"    - eval samples: {len(eval_dataset)}")

    # Base training hyper-parameters (shared with dpo_trainer.py)
    dpo_kwargs = dict(
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
        fp16=config.get("fp16", False),
        tf32=config.get("tf32", False),
        eval_strategy=eval_strategy,
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        max_prompt_length=config.get("max_prompt_length", 512),
        max_length=config.get("max_length", 1024),
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    if eval_strategy != "no":
        dpo_kwargs["eval_steps"] = config.get("eval_steps", config["save_steps"])
        dpo_kwargs["per_device_eval_batch_size"] = config.get(
            "per_device_eval_batch_size", config["per_device_train_batch_size"]
        )

    mdpo_args = DPOConfig(**dpo_kwargs)

    # Initialize the MDPO trainer
    trainer = MDPOTrainer(
        model=model,
        args=mdpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        gamma=config.get("gamma", 0.9),
        K=config.get("K", 12),
    )

    if use_lora and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    print("[*] Starting MDPO training...")
    trainer.train()

    final_save_path = os.path.join(config["output_dir"], config.get("output_model_name", "final_model"))
    trainer.save_model(final_save_path)
    print(f"Training finished. Model saved to: {final_save_path}")

if __name__ == "__main__":
    main()

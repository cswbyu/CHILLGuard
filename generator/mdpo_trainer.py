import os
import json
import torch
import torch.nn.functional as F
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
    def __init__(self, *args, gamma=0.9, K=12, **kwargs):
        # Extract MDPO hyper-parameters from the training config
        super().__init__(*args, **kwargs)
        self.beta_base = self.args.beta
        self.gamma = gamma
        self.K = K
        self.running_reward_gap = torch.tensor(
            1.0, device=self.args.device
        )
        self.current_dynamic_beta = self.beta_base

    @staticmethod
    def _per_token_logps(logits, labels):
        """log softmax(logits)[labels], computed row by row for half precision to bound memory."""
        if logits.dtype in (torch.float32, torch.float64):
            return torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        rows = []
        for row_logits, row_labels in zip(logits, labels):
            row_logps = F.log_softmax(row_logits, dim=-1)
            rows.append(row_logps.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1))
        return torch.stack(rows)

    def _sequence_logps(self, model, inputs):
        """log pi(y|x) (Eq. 1): prompt x conditions the model; log-probs summed over response tokens y only.
        TRL (>= 0.29) collates input_ids = [x + y_w ; x + y_l] with `completion_mask` marking y."""
        input_ids = inputs["input_ids"]
        logits = model(input_ids=input_ids, attention_mask=inputs["attention_mask"], use_cache=False).logits
        per_token_logps = self._per_token_logps(logits[:, :-1, :], input_ids[:, 1:])
        response_mask = inputs["completion_mask"][:, 1:].to(per_token_logps.dtype)
        return (per_token_logps * response_mask).sum(-1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Core MDPO loss (paper Sec. 4.2, Eq. 1-7). B is the per-device batch."""
        policy_chosen_logps, policy_rejected_logps = self._sequence_logps(model, inputs).chunk(2, dim=0)
        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_logps = self._sequence_logps(self.ref_model, inputs)
                else:  # LoRA: reference = base model with the adapter disabled
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_logps = self._sequence_logps(model, inputs)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        logits_diff = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
        reward_gaps = self.beta_base * logits_diff                                   # Eq. 3
        global_mean = self.running_reward_gap.detach().to(reward_gaps.device)        # R_bar
        dist_sq = (reward_gaps.detach() - global_mean) ** 2                          # Eq. 4
        keep_idx = torch.topk(dist_sq, k=min(self.K, reward_gaps.size(0)), largest=False).indices
        batch_avg_gap = reward_gaps[keep_idx].mean()                                 # Eq. 5
        alpha_m = torch.sigmoid(batch_avg_gap) / torch.sigmoid(global_mean)          # Eq. 6
        dynamic_beta = self.beta_base * alpha_m
        self.current_dynamic_beta = dynamic_beta.item()
        losses = -F.logsigmoid(dynamic_beta * logits_diff)                           # Eq. 2 with beta_M
        if model.training:                                                           # Eq. 7
            self.running_reward_gap = self.gamma * global_mean + (1 - self.gamma) * batch_avg_gap.detach()
        loss = losses.mean()
        return (loss, {}) if return_outputs else loss

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

    # Base training hyper-parameters (shared with the static-beta DPO baseline dpo_lora_trainer.py)
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

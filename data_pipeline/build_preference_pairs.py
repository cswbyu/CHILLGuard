import argparse
import json
import logging
import re
import os
import multiprocessing as mp
import concurrent.futures
from collections import defaultdict
from pathlib import Path

# Shared PE rewriting templates (single source of truth in templates.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates import get_prompts

'''
Build MDPO preference pairs (paper Sec. 4.3, Eq. 8-9):
the current classifier labels every generated prompt while the generator rates it,
the two signals are mapped to difficulty levels L1-L4, and preference pairs are
formed following the priority rule. Level-1 (hard) samples are also exported as
incremental training data for the next classifier update.

python build_preference_pairs.py \
  --input_file ./data/iter1_gen.jsonl \
  --classifier_model ./checkpoints/classifier_iter1 \
  --generator_model ./models/Dolphin3.0-Llama3.1-8B \
  --output_dir ./data/iter1_pairs \
  --status iter1 \
  --cls_gpu 3 \
  --gen_gpu 1 \
  --gpu_memory_utilization 0.6
'''


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REFUSAL_PHRASES = [
    "i apologize", "i cannot comply", "i can't fulfill", "as an ai",
    "我无法", "我不能", "很抱歉", "作为一个人工智能", "无法满足"
]

cur_status = "iter1"  # set from --status in main()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parallel CHILLGuard Data Pipeline")
    parser.add_argument('--input_file', type=str, required=True, help='Raw generated-data JSONL file')
    parser.add_argument('--classifier_model', type=str, required=True, help='Classifier checkpoint path')
    parser.add_argument('--generator_model', type=str, required=True, help='Generator / rater model path')
    parser.add_argument('--output_dir', type=str, default='outputs/next_iteration_data', help='Output directory')
    parser.add_argument('--status', type=str, default='iter1', help='Iteration tag used to name the output files (e.g. iter1, iter2)')
    parser.add_argument('--cls_gpu', type=str, default="0", help='GPU ID(s) for the classifier, single "0" or multiple "0,1"')
    parser.add_argument('--gen_gpu', type=str, default="1", help='GPU ID(s) for the generator / rater')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_length_diff', type=int, default=500, help='Maximum character-length difference kept between original and rewritten prompt')
    return parser.parse_args()


def parse_gpu_ids(gpu_ids: str) -> list:
    ids = [x.strip() for x in str(gpu_ids).split(',') if x.strip()]
    if not ids:
        raise ValueError("GPU ID list cannot be empty")
    if not all(x.isdigit() for x in ids):
        raise ValueError(f"Invalid GPU ID format: {gpu_ids}")
    return ids

# Standard chat format for the guardrail classifier
def format_for_classifier(text: str) -> list:
    return [
        {"role": "system", "content": "你是一个安全审核助手。请只输出'安全'或'不安全'。"},
        {"role": "user", "content": f"请判断以下文本是否安全：\n{text}"}
    ]

# Fully-Chinese chat format for the generator G(0) acting as a quality rater (paper Eq. 8)
def format_for_rating(text: str, language="中文") -> list:
    return [
        {"role": "system", "content": f"你是一个{language}文本质量评估专家。你的任务是评估提示词的生成质量。"},
        {"role": "user", "content": f"请评估以下#{language}提示词#的生成质量，并给出1到5的评分（1=质量很差，5=质量很好）。\n请务必在回复的第一行仅输出一个阿拉伯数字评分。\n\n#提示词#:\n{text}"}
    ]

# ==========================================
# Worker 1 (separate process): classifier inference
# ==========================================
def run_classifier_task(model_path, prompts, gpu_id, mem_util):
    gpu_ids = parse_gpu_ids(gpu_id)
    visible_devices = ",".join(gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    from vllm import LLM, SamplingParams

    tensor_parallel_size = len(gpu_ids)
    logging.info(f"[Classifier] Initializing on GPU {visible_devices}, tensor_parallel_size={tensor_parallel_size}...")
    llm = LLM(
        model=model_path,
        max_model_len=20000,
        gpu_memory_utilization=mem_util,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    # Apply the model's official chat template so the system prompt is recognized
    tokenizer = llm.get_tokenizer()
    chat_prompts = []
    for p in prompts:
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            kw["enable_thinking"] = False
            chat_prompts.append(tokenizer.apply_chat_template(p, **kw))
        except TypeError:  # templates without a thinking section ignore the flag
            del kw["enable_thinking"]
            chat_prompts.append(tokenizer.apply_chat_template(p, **kw))

    logging.info("[Classifier] Running inference...")
    outputs = llm.generate(chat_prompts, sampling_params)

    predictions = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        # Tolerant parsing: any mention of "unsafe"/"dangerous" means a negative label
        if "不安全" in text or "危险" in text:
            predictions.append("不安全")
        else:
            predictions.append("安全")
    return predictions

# ==========================================
# Worker 2 (separate process): rater inference
# ==========================================
def run_evaluator_task(model_path, prompts, gpu_id, mem_util):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams
    import re

    logging.info(f"[Evaluator] Initializing on GPU {gpu_id}...")
    llm = LLM(model=model_path, max_model_len=8192, gpu_memory_utilization=mem_util, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    # Apply the model's official chat template
    tokenizer = llm.get_tokenizer()
    chat_prompts = []
    for p in prompts:
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            kw["enable_thinking"] = False
            chat_prompts.append(tokenizer.apply_chat_template(p, **kw))
        except TypeError:  # templates without a thinking section ignore the flag
            del kw["enable_thinking"]
            chat_prompts.append(tokenizer.apply_chat_template(p, **kw))

    logging.info("[Evaluator] Running inference...")
    outputs = llm.generate(chat_prompts, sampling_params)

    scores = []
    for out in outputs:
        text = out.outputs[0].text.strip().split("\n")[0]
        try:
            scores.append(int(re.findall(r'\d+', text)[0]))
        except:
            scores.append(1)  # unratable responses default to the lowest quality score
    return scores

def has_refusal(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in REFUSAL_PHRASES)

def determine_level(original_label: str, eval_score: int, classifier_pred: str) -> int:
    """Map (gold label, generator quality score, classifier prediction) to the
    difficulty levels L1-L4 of paper Eq. 8.

    L1: misclassified + eval_score >= 3
    L2: misclassified + eval_score < 3
    L3: correctly classified + eval_score < 3
    L4: correctly classified + eval_score >= 3
    """
    is_safe_label = (original_label == "安全")
    is_safe_pred = (classifier_pred == "安全")
    is_misclassified = (is_safe_label != is_safe_pred)

    if is_misclassified and eval_score >= 3: return 1
    elif is_misclassified: return 2
    elif eval_score < 3: return 3
    else: return 4

def build_dpo_prompt(seed_text: str, label: str) -> str:
    is_safe = (label == "安全")
    system_prompt, user_prompt = get_prompts(seed_text, is_safe)
    return (
        f"系统：\n{system_prompt}\n\n"
        f"用户：\n{user_prompt}\n\n"
        "助手："
    )

def main():
    global cur_status
    args = parse_arguments()
    cur_status = args.status
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading raw data: {args.input_file}")
    raw_data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    aug_prompts = [item.get("augmented_prompt", "") for item in raw_data]
    cls_prompts = [format_for_classifier(p) for p in aug_prompts]
    eval_prompts = [format_for_rating(p, language="中文") for p in aug_prompts]

    # ==========================================
    # Run both inference tasks concurrently in separate processes
    # ==========================================
    ctx = mp.get_context('spawn')
    logging.info("Launching concurrent inference (Classifier & Evaluator)...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
        future_cls = executor.submit(
            run_classifier_task, args.classifier_model, cls_prompts, args.cls_gpu, args.gpu_memory_utilization
        )
        future_eval = executor.submit(
            run_evaluator_task, args.generator_model, eval_prompts, args.gen_gpu, args.gpu_memory_utilization
        )

        predictions = future_cls.result()
        scores = future_eval.result()

    # ==========================================
    # Cleaning, difficulty rating and pair construction
    # ==========================================
    logging.info("Inference done; cleaning data and building preference pairs...")
    grouped_candidates = defaultdict(list)

    for i, item in enumerate(raw_data):
        item["classifier_pred"] = predictions[i]
        item["eval_score"] = scores[i]

        orig_p = item.get("original_prompt", "")
        aug_p = item.get("augmented_prompt", "")

        if abs(len(orig_p) - len(aug_p)) > args.max_length_diff: continue
        if has_refusal(aug_p): continue

        item["level"] = determine_level(item["label"], item["eval_score"], item["classifier_pred"])
        grouped_candidates[item["id"]].append(item)

    classifier_new_data, dpo_pairs = [], []

    for seed_id, candidates in grouped_candidates.items():
        levels = defaultdict(list)
        for cand in candidates:
            levels[cand["level"]].append(cand)
            if cand["level"] == 1:
                classifier_new_data.append(cand)

        # Priority rule of paper Eq. 9: <L1,L4> > <L1,L3> > <L2,L4>.
        # Preference pairs require a chosen/rejected candidate from the listed
        # level combinations; seeds without such a combination yield no pair.
        preferred = dispreferred = None
        if levels[1] and levels[4]:
            preferred, dispreferred = levels[1][0], levels[4][0]
        elif levels[1] and levels[3]:
            preferred, dispreferred = levels[1][0], levels[3][0]
        elif levels[2] and levels[4]:
            preferred, dispreferred = levels[2][0], levels[4][0]

        if preferred is None or dispreferred is None:
            continue

        dpo_prompt = build_dpo_prompt(preferred["original_prompt"], preferred["label"])
        dpo_pairs.append({
            "prompt": dpo_prompt,
            "seed_prompt": preferred["original_prompt"],
            "chosen": preferred["augmented_prompt"],
            "rejected": dispreferred["augmented_prompt"],
            "original_label": preferred["label"]
        })

    # ==========================================
    # Persist outputs
    # ==========================================
    all_scored_out = output_dir / f"{cur_status}_build_log.jsonl"
    with open(all_scored_out, 'w', encoding='utf-8') as f:
        for item in raw_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(output_dir / f"{cur_status}_level1.jsonl", 'w', encoding='utf-8') as f:
        for item in classifier_new_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(output_dir / f"{cur_status}_dpo_pairs.jsonl", 'w', encoding='utf-8') as f:
        for item in dpo_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logging.info(f"Pipeline finished!")
    logging.info(f"Saved {len(raw_data)} fully scored samples to: {cur_status}_build_log.jsonl")
    logging.info(f"Saved {len(classifier_new_data)} classifier-update samples and {len(dpo_pairs)} DPO pairs.")

if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="lora_sft"
# Working directory defaults to this script's directory (training code); override via env var
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Conda environment is machine-specific; override via env vars
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-chillguard}"
LOCK_FILE="$WORKDIR/.lora_sft.lock"
read -r -a GPU_IDS <<< "${GPU_IDS:-0 1}"
MAX_WAIT_SECONDS=7200
CHECK_INTERVAL_SECONDS=30

# Treat memory/utilization thresholds as "idle enough" for starting a new run.
MAX_MEM_MB=500
MAX_UTIL=10

timestamp() {
	date "+%F %T"
}

gpu_idle() {
	local gid="$1"
	local stats mem util pids

	stats=$(nvidia-smi --id="$gid" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
	mem="${stats%%,*}"
	util="${stats##*,}"

	# If there are compute pids, the GPU is not idle.
	pids=$(nvidia-smi --id="$gid" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
	if echo "$pids" | grep -qE '[0-9]'; then
		return 1
	fi

	[[ "$mem" -lt "$MAX_MEM_MB" && "$util" -lt "$MAX_UTIL" ]]
}

all_target_gpus_idle() {
	local gid
	for gid in "${GPU_IDS[@]}"; do
		if ! gpu_idle "$gid"; then
			return 1
		fi
	done
	return 0
}

launch_training_in_tmux() {
	local log_file gpu_csv launch_cmd
	gpu_csv=$(IFS=,; echo "${GPU_IDS[*]}")
	log_file="$WORKDIR/lora_sft_$(date +%F_%H-%M-%S).log"

	launch_cmd="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$WORKDIR\" && CUDA_VISIBLE_DEVICES=$gpu_csv accelerate launch --num_processes=2 lora_sft_trainer.py --config ./configs/sft_config.json --multi_gpu 2>&1 | tee -a \"$log_file\""

	if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
		echo "[$(timestamp)] tmux session '$SESSION_NAME' already exists. Skip launch."
		return 0
	fi

	tmux new-session -d -s "$SESSION_NAME" "bash -lc '$launch_cmd'"
	echo "[$(timestamp)] launched in tmux session '$SESSION_NAME'"
	echo "[$(timestamp)] log file: $log_file"
}

main() {
	exec 9>"$LOCK_FILE"
	if ! flock -n 9; then
		echo "[$(timestamp)] lock busy, another launcher instance is running."
		exit 0
	fi

	local waited=0
	while (( waited <= MAX_WAIT_SECONDS )); do
		if all_target_gpus_idle; then
			launch_training_in_tmux
			exit 0
		fi
		echo "[$(timestamp)] GPUs ${GPU_IDS[*]} busy, retry after ${CHECK_INTERVAL_SECONDS}s..."
		sleep "$CHECK_INTERVAL_SECONDS"
		waited=$(( waited + CHECK_INTERVAL_SECONDS ))
	done

	echo "[$(timestamp)] timeout (${MAX_WAIT_SECONDS}s). Did not launch training."
	exit 1
}

main "$@"

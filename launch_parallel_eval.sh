#!/bin/bash
CONFIG="configs/config_baseline.yaml"
# LIMIT="[0,1,2,3,4,5,6,7,8,9]" 
LIMIT=None
# GPUS=(0 1 2 3 4 5 6 7) 
# GPUS=(4 5 6 7) 
GPUS=(2 3)
PYTHON=python
SCRIPT="batch_eval_run.py"
LOG_DIR="/data/yuansheng/eval_results/logs/baseline/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# CHECKPOINTS=(
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-250"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-500"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-750"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-1000"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-1250"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-1500"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-1750"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-2000"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-2250"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-2500"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-2750"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-3000"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-3250"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-3500"
#   "/data/yuansheng/checkpoint/qwen2_5_7b_coder_stage4_lr5e6/v0-20250505-221618/checkpoint-3750"
# )

# CHECKPOINTS=(
#   "/data/yuansheng/checkpoint/qwen3_4b_stage4_lr5e6/epoch1"
#   "/data/yuansheng/checkpoint/qwen3_4b_stage4_lr5e6/epoch2"
#   "/data/yuansheng/checkpoint/qwen3_4b_stage4_lr5e6/epoch3"
# )

# CHECKPOINTS=(
#   "Qwen/Qwen2.5-7B-Instruct"
#   "Qwen/Qwen2.5-Coder-7B-Instruct"
#   "meta-llama/Llama-3.1-8B-Instruct"
#   "meta-llama/Llama-3.2-3B-Instruct"
#   "meta-llama/Llama-3.2-1B-Instruct"
# )

# CHECKPOINTS=(
#   "Qwen/Qwen3-4B"
#   "Qwen/Qwen3-8B"
# )

CHECKPOINTS=(
  "openai/gpt-4o"
  "openai/gpt-4o-mini"
)

GPU_COUNT=${#GPUS[@]}
TOTAL=${#CHECKPOINTS[@]}

for ((i=0; i<TOTAL; i+=GPU_COUNT)); do
  echo "[INFO] Starting batch from checkpoint $i..."
  for ((j=0; j<GPU_COUNT && i+j<TOTAL; j++)); do
    GPU_ID=${GPUS[$j]}
    CKPT=${CHECKPOINTS[$((i+j))]}
    LOG_FILE="$LOG_DIR/eval_$(basename $CKPT).log"

    echo "[INFO] Launching on GPU $GPU_ID -> $CKPT"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $SCRIPT \
      --cuda=$GPU_ID \
      --model=$CKPT \
      --limit=$LIMIT \
      --config=$CONFIG \
      > "$LOG_FILE" 2>&1 &

    sleep 2 
  done

  wait 
done

echo "[INFO] ✅ All evaluation jobs completed."

#!/bin/bash
CONFIG="configs/config_baseline.yaml"
# LIMIT="[0,1,2,3,4,5,6,7,8,9]" 
LIMIT=None
# GPUS=(0 1 2 3 4 5 6 7) 
# GPUS=(4 5 6 7) 
GPUS=(7)
PYTHON=python
SCRIPT="batch_eval_run.py"
LOG_DIR="/mnt/tjena/yuansheng/PandasPlotBench/eval_results/logs/baseline/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CHECKPOINTS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
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

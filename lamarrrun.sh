#!/bin/bash
#SBATCH --job-name=swinmars
#SBATCH --time=72:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24
#SBATCH --output=out/swinmars-%j.out
#SBATCH --error=out/swinmars-%j.err

# 检查并创建输出目录
OUT_DIR="out"
if [ ! -d "$OUT_DIR" ]; then
    mkdir -p "$OUT_DIR"
fi
# 获取当前日期和时间
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 获取当前时间的秒数
START_SECONDS=$(date +%s)

# 设置日志文件路径
LOG_FILE="jilulog/experiment_$CURRENT_TIME.txt"

# 检查并创建日志文件目录
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# 记录开始时间和命令
echo "Experiment started at $CURRENT_TIME" >> "$LOG_FILE"

# 执行命令并将stdout追加到日志文件，同时输出到终端（保留控制字符）
function execute_and_log {
    local CMD="$1"
    echo "Executing: $CMD" >> "$LOG_FILE"
    eval "$CMD" | while IFS= read -r line; do
        # 计算从开始到现在所用的时间
        CURRENT_SECONDS=$(date +%s)
        ELAPSED_SECONDS=$((CURRENT_SECONDS - START_SECONDS))
        ELAPSED_MINUTES=$((ELAPSED_SECONDS / 60))
        ELAPSED_SECONDS=$((ELAPSED_SECONDS % 60))
        # 添加时间信息并输出
        printf "[%d:%02d] %s\n" "$ELAPSED_MINUTES" "$ELAPSED_SECONDS" "$line"
    done | tee -ai "$LOG_FILE"
}


# 在后台运行 nvidia-smi 命令并将输出重定向到日志文件
(sleep 300 && nvidia-smi) >> "$LOG_FILE" 2>&1 &

/home/ma1/anaconda3/envs/vitnew/bin/python -u train.py --config_file \
configs/mars/swin_base.yml \
MODEL.PRETRAIN_CHOICE 'self' \
MODEL.PRETRAIN_PATH 'pretrained_model/checkpoint_tea.pth' \
OUTPUT_DIR './log/mars/swin_base' \
SOLVER.BASE_LR 0.0002 \
SOLVER.OPTIMIZER_NAME 'SGD' \
MODEL.SEMANTIC_WEIGHT 0.2 \
SOLVER.IMS_PER_BATCH 384 \
SOLVER.MAX_EPOCHS 31 \
TEST.IMS_PER_BATCH 2000 \
DATALOADER.NUM_WORKERS 24 \
SOLVER.EVAL_PERIOD 30 \
SOLVER.CHECKPOINT_PERIOD 29 \
MODEL.DIST_TRAIN True


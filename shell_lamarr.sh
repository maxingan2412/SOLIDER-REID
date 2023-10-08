#!/bin/bash

# sh xxx.sh trainlog
# 检查是否有参数传递给脚本
if [ "$#" -ne 1 ]; then
    echo "使用方法: $0 <日志名称>"
    exit 1
fi

# 获取日志名称参数
LOG_NAME=$1

# 设置日志文件夹名称（你可以根据需要修改）
LOG_FOLDER="shell_lamar_log"

# 创建日志文件夹，如果不存在
mkdir -p "$LOG_FOLDER"

# 获取当前时间并格式化为文件名
CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_FILE="./${LOG_FOLDER}/${LOG_NAME}_${CURRENT_TIME}.txt"

# 使用nohup在后台执行训练命令，并将所有输出（包括错误输出）重定向到日志文件
nohup python train.py \
--config_file configs/mars/swin_base.yml \
MODEL.PRETRAIN_CHOICE 'self' \
MODEL.PRETRAIN_PATH 'pretrained_model/checkpoint_tea.pth' \
OUTPUT_DIR './log/mars/swin_base' \
SOLVER.BASE_LR 0.0002 \
SOLVER.OPTIMIZER_NAME 'SGD' \
SOLVER.CHECKPOINT_PERIOD 40 \
SOLVER.EVAL_PERIOD 30 \
MODEL.SEMANTIC_WEIGHT 0.2 \
SOLVER.IMS_PER_BATCH 96 \
DATALOADER.NUM_WORKERS 24 \
>> "$LOG_FILE" 2>&1 &

# 将脚本结果写入日志文件
echo "训练命令已在后台启动。训练日志文件已保存到：$LOG_FILE"

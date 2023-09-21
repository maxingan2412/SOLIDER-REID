#!/bin/bash

# 设置日志文件夹名称（你可以根据需要修改）
LOG_FOLDER="my_logs"

# 创建日志文件夹，如果不存在
mkdir -p "$LOG_FOLDER"

# 设置环境变量
export RANK=0  # 设置当前进程的排名
export WORLD_SIZE=1  # 设置总的进程数
export MASTER_ADDR=localhost  # 设置主节点地址
export MASTER_PORT=12345  # 设置主节点端口

# 获取当前时间并格式化为文件名
CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_FILE="./${LOG_FOLDER}/train_log_${CURRENT_TIME}.txt"

# 执行训练命令并将输出重定向到日志文件，并且不显示在终端上，同时在后台运行
python train.py --config_file \
configs/mars/swin_base.yml \
MODEL.PRETRAIN_CHOICE 'self' \
MODEL.PRETRAIN_PATH 'pretrained_model/checkpoint_tea.pth' \
OUTPUT_DIR "./log/mars/swin_base" \
SOLVER.BASE_LR 0.0002 \
SOLVER.OPTIMIZER_NAME 'SGD' \
MODEL.SEMANTIC_WEIGHT 0.2 \
SOLVER.IMS_PER_BATCH 384 \
SOLVER.MAX_EPOCHS 31 \
TEST.IMS_PER_BATCH 2000 \
DATALOADER.NUM_WORKERS 24 \
SOLVER.EVAL_PERIOD 1 \
SOLVER.CHECKPOINT_PERIOD 29 \
MODEL.DEVICE_ID 0,1 \
MODEL.DIST_TRAIN True > "$LOG_FILE" 2>&1 &

# 输出日志文件路径
echo "训练日志文件已保存到：$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "训练成功！"
else
    echo "训练失败，请查看日志文件以获取详细信息：$LOG_FILE"
fi


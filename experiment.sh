#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <log_filename_without_extension> [command_to_run ...]"
    exit 1
fi

LOG_FILENAME="$1"
shift

# 获取当前日期和时间
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 获取当前时间的秒数
START_SECONDS=$(date +%s)

# 设置日志文件路径
LOG_FILE="2024july25/${LOG_FILENAME}_$CURRENT_TIME.txt"

# 检查并创建日志文件目录
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# 记录开始时间和命令
echo "Experiment started at $CURRENT_TIME" >> "$LOG_FILE"
echo "Command: $*" >> "$LOG_FILE"

# 后台执行命令并将stdout和stderr记录到日志文件
{
    "$@" > >(while IFS= read -r line; do
        # 计算从开始到现在所用的时间
        CURRENT_SECONDS=$(date +%s)
        ELAPSED_SECONDS=$((CURRENT_SECONDS - START_SECONDS))
        ELAPSED_MINUTES=$((ELAPSED_SECONDS / 60))
        ELAPSED_SECONDS=$((ELAPSED_SECONDS % 60))
        # 添加时间信息并输出到日志文件
        printf "[%d:%02d] %s\n" "$ELAPSED_MINUTES" "$ELAPSED_SECONDS" "$line" >> "$LOG_FILE"
    done) 2> >(while IFS= read -r line; do
        # 计算从开始到现在所用的时间
        CURRENT_SECONDS=$(date +%s)
        ELAPSED_SECONDS=$((CURRENT_SECONDS - START_SECONDS))
        ELAPSED_MINUTES=$((ELAPSED_SECONDS / 60))
        ELAPSED_SECONDS=$((ELAPSED_SECONDS % 60))
        # 添加时间信息并输出到日志文件
        printf "[%d:%02d] %s\n" "$ELAPSED_MINUTES" "$ELAPSED_SECONDS" "$line" >> "$LOG_FILE"
    done)

    # 记录结束时间
    echo "Experiment ended at $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
} &

# 记录 PID
COMMAND_PID=$!
echo "PID: $COMMAND_PID" >> "$LOG_FILE"

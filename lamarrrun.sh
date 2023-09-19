#!/bin/bash
#SBATCH --job-name=swin_mars
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=out/swin_mars-%j.out
#SBATCH --error=out/swin_mars-%j.err


/home/ma1/anaconda3/envs/vitnew/bin/python -u train.py --config_file \
configs/mars/swin_base.yml \
MODEL.PRETRAIN_CHOICE 'self' \
MODEL.PRETRAIN_PATH 'pretrained_model/checkpoint_tea.pth' \
OUTPUT_DIR './log/mars/swin_base' \
SOLVER.BASE_LR 0.0002 \
SOLVER.OPTIMIZER_NAME 'SGD' \
MODEL.SEMANTIC_WEIGHT 0.2 \
SOLVER.IMS_PER_BATCH 384 \
SOLVER.MAX_EPOCHS 120 \
TEST.IMS_PER_BATCH 2000 \
DATALOADER.NUM_WORKERS 12 \
SOLVER.EVAL_PERIOD 120 \
SOLVER.CHECKPOINT_PERIOD 59


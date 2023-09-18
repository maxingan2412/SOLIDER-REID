CUDA_VISIBLE_DEVICES=0 python test.py --config_file \
configs/market/swin_base.yml \
TEST.WEIGHT './log/market1501/swin_base/swin_base_market.pth' \
TEST.RE_RANKING True \
MODEL.SEMANTIC_WEIGHT 0.2

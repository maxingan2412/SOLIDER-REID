import logging
import os.path as osp
import sys
import os
from datetime import datetime


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # 获取当前时间并格式化
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

        if if_train:
            log_filename = "{}_train_log.txt".format(current_time)
        else:
            log_filename = "{}_test_log.txt".format(current_time)

        fh = logging.FileHandler(os.path.join(save_dir, log_filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

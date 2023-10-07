from utils.logger import setup_logger
from datasets import make_dataloader,make_mars_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss, make_mars_loss
from processor import do_train, do_mars_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import torch.distributed as dist

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR)) #这个可以在终端打印
    #logger.info("相关参数".format(args))

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Running with config:\n{}".format(cfg))


    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    if cfg.DATASETS.NAMES == 'mars':
        train_loader, num_query, num_classes, camera_num, view_num, q_val_set, g_val_set = make_mars_dataloader(cfg)
    else:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

        # train_loader, num_query, num_classes, camera_num, view_num, q_val_set, g_val_set = make_mars_dataloader(cfg.DATASETS.NAMES,
        #                                                                                               cfg.SOLVER.IMS_PER_BATCH,
        #                                                                                               4, #seqlen
        #                                                                                               cfg.DATALOADER.NUM_WORKERS)  # 这里完成了 datloader的组合

    #train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)

    # if cfg.DATASETS.NAMES == 'mars':
    #     loss_func, center_criterion = make_mars_loss(cfg, num_classes=num_classes)
    # else:
    #     loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)


    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)



    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)


    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    if cfg.DATASETS.NAMES == 'mars':
        do_mars_train(cfg, model, center_criterion, train_loader, q_val_set,g_val_set,optimizer, optimizer_center, scheduler, loss_func, num_query, args.local_rank)
    else:
        do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_func, num_query, args.local_rank)




    # #do_train(
    #     cfg,
    #     model,
    #     center_criterion,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     optimizer_center,
    #     scheduler,
    #     loss_func,
    #     num_query, args.local_rank
    # )

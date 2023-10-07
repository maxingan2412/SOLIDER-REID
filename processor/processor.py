import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from tqdm import tqdm

from datetime import datetime
from torch.nn import functional as F
#from utils.reranking import re_ranking

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1)
        ],
        axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(
        1. * original_dist / np.max(original_dist, axis=0)
    )
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate
            )[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]
            if len(
                np.
                intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2.-temp_min)

    final_dist = jaccard_dist * (1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat



def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
    return distmat
def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       # >>> from torchreid import metrics
       # >>> input1 = torch.rand(10, 2048)
       # >>> input2 = torch.rand(100, 2048)
       # >>> distmat = metrics.compute_distance_matrix(input1, input2)
       # >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat
def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def extract_features(data_loader, model, use_gpu=True, pool='avg'):
    features_list, pids_list, camids_list = [], [], []
    with (torch.no_grad()):
        for batch_idx, data in enumerate(tqdm(data_loader)):
            data = data[0]
            imgs, pids, camids,_ = data

            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)

            b, s, c, h, w = imgs.size()
            features, _  = model(imgs, pids, cam_label=camids)
            features = features.view(b, -1)
            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)
            features = features.data.cpu()

            features_list.append(features)

            # Ensure pids and camids are iterable (list or tensor) before extending
            if not isinstance(pids, (list, torch.Tensor)):
                pids = [pids]
            if not isinstance(camids, (list, torch.Tensor)):
                camids = [camids]

            pids_list.extend(pids)
            camids_list.extend(camids)



    #823这里的提取特征是一个tracklets提出来，每一个tracklets对应了一个feature
    features = torch.stack(features_list) # 一个1980场的list 变成  1980 * 13506的tensro
    pids = np.asarray(pids_list) #1980的list 变为array
    camids = np.asarray(camids_list) #110860

    return features, pids, camids

def test_mars(model, queryloader, galleryloader, pool='avg', use_gpu=True):
    model.eval()
    #qf1980 13056 tensor  q_pids 1980  q_camids 110860 因为一个tracklets id一样，所以是1980，但是camerid不一定一样 所以每个tacklets下的每个图片的camid都得记录，是110860
    qf, q_pids, q_camids = extract_features(queryloader, model, use_gpu, pool)
    gf, g_pids, g_camids = extract_features(galleryloader, model, use_gpu, pool)

    print(f"Extracted features for query set, obtained {qf.size(0)}-by-{qf.size(1)} matrix")
    print(f"Extracted features for gallery set, obtained {gf.size(0)}-by-{gf.size(1)} matrix")

    print("Computing distance matrix")
    #distmat = compute_distance_matrix(qf, gf).numpy()
    metricchoose = 'euclidean'
    distmat = compute_distance_matrix(qf, gf, metricchoose)
    distmat = distmat.numpy()

    rerank = True
    if rerank:
        #print('Applying person re-ranking ...')
        distmat_qq = compute_distance_matrix(qf, qf, metricchoose)
        distmat_gg = compute_distance_matrix(gf, gf, metricchoose)
        distmat1 = re_ranking(distmat, distmat_qq, distmat_gg)
        print("Computing CMC and mAP with reranking,")
        cmc, mAP = evaluate(distmat1, q_pids, g_pids, q_camids, g_camids)

        print("Results ---------- ")
        print(f"mAP: {mAP:.1%}")
        print(f"CMC curve r1: {cmc[0]}")






    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ---------- ")
    print(f"mAP: {mAP:.1%}")
    print(f"CMC curve r1: {cmc[0]}")

    return cmc[0], mAP



def model_inference(model, device, val_loader, evaluator, cfg):
    logger = logging.getLogger("transreid.test")
    model.eval()
    for n_iter, (img, vid, camid, camids, target_view, *args) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat, _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    torch.cuda.empty_cache()
    return cmc, mAP






def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        #
        for n_iter, (img, vid, target_cam, target_view) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device) # bs 3,h,w tensor
            target = vid.to(device) # bs tensor
            target_cam = target_cam.to(device) # bs tensor
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list): #这里的acc也就是 这一个batch中 我用模型得到的id和实际的id的准确率
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            # Ensure directory exists; if not, create it
            if not os.path.exists(cfg.OUTPUT_DIR):
                os.makedirs(cfg.OUTPUT_DIR)

            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(tqdm(val_loader)):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, _ = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(tqdm(val_loader)):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, _ = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_mars_train(cfg,
             model,
             center_criterion,
             train_loader,
             q_val_set,
             g_val_set,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        #
        for n_iter, (img, pid, target_cam, labels2) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device) # bs 3,h,w tensor
            target = pid.to(device) # bs tensor
            target_cam = target_cam.to(device) # bs tensor
            labels2 = labels2.to(device)
            with amp.autocast(enabled=True):
                target_cam = target_cam.view(-1)
                score, feat, _ = model(img, label=target, cam_label=target_cam)

                #为了适应 swin的形状 给 target变成4倍
                #target = torch.repeat_interleave(target, repeats=4)
                #target = target.repeat(4)

                loss = loss_fn(score, feat, target, target_cam) #实际没用上 target_cam，vid里面的loss也没用和这个

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list): #这里的acc也就是 这一个batch中 我用模型得到的id和实际的id的准确率
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            # Ensure directory exists; if not, create it
            if not os.path.exists(cfg.OUTPUT_DIR):
                os.makedirs(cfg.OUTPUT_DIR)

            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            # if cfg.MODEL.DIST_TRAIN:
            #     if dist.get_rank() == 0:
            #         model.eval()
            #         for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(tqdm(val_loader)):
            #             with torch.no_grad():
            #                 img = img.to(device)
            #                 camids = camids.to(device)
            #                 target_view = target_view.to(device)
            #                 feat, _ = model(img, cam_label=camids, view_label=target_view)
            #                 evaluator.update((feat, vid, camid))
            #         cmc, mAP, _, _, _, _, _ = evaluator.compute()
            #         logger.info("Validation Results - Epoch: {}".format(epoch))
            #         logger.info("mAP: {:.1%}".format(mAP))
            #         for r in [1, 5, 10]:
            #             logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            #         torch.cuda.empty_cache()
            # else:
                # model.eval()
                # for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(tqdm(val_loader)):
                #     with torch.no_grad():
                #         img = img.to(device)
                #         camids = camids.to(device)
                #         target_view = target_view.to(device)
                #         feat, _ = model(img, cam_label=camids, view_label=target_view)
                #         evaluator.update((feat, vid, camid))
                # cmc, mAP, _, _, _, _, _ = evaluator.compute()
                # logger.info("Validation Results - Epoch: {}".format(epoch))
                # logger.info("mAP: {:.1%}".format(mAP))
                # for r in [1, 5, 10]:
                #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                # torch.cuda.empty_cache()

            model.eval()
            cmc, map = test_mars(model, q_val_set, g_val_set)
            print('CMC: %.4f, mAP : %.4f' % (cmc, map))
                # if cmc_rank1 < cmc:
                #    cmc_rank1=cmc

                # save_path = 'VID-Trans-ReID_pth'
                # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                #
                # file_name = f"{cfg.DATASETS.NAMES}_BS{cfg.SOLVER.IMS_PER_BATCH}_Epoch{epoch}_CMC{cmc:.4f}_MAP{map:.4f}_{current_time}.pth"
                # save_filename = os.path.join(save_path, file_name)
                #
                # # 创建目录，如果它不存在
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)
                # torch.save(model.state_dict(), save_filename)





def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat , _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_mars_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat , _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
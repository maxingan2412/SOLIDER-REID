import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224,pose_swin_base_patch4_window7_224
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth
import random
#import faiss
from kmeans_pytorch import kmeans

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf

    code from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """

        # (N, T, H) -> (N, T) -> (N, T, 1)
        att_w = nn.functional.softmax(self.W(x).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x


# def cluster_tensor_on_gpu(tensor, n_clusters):
#     """
#     使用FAISS GPU版本对形状为(b, s, c, h, w)的张量进行聚类。
#
#     参数:
#     - tensor: 输入的形状为(b, s, c, h, w)的张量。
#     - n_clusters: 聚类的数量。
#
#     返回:
#     - centroids: 聚类的中心点。
#     - assignments: 每个特征向量的聚类分配。
#     """
#
#     # 确保张量在CPU上，并使用detach()来确保它不需要梯度
#     tensor = tensor.detach().cpu().numpy()
#
#     # 将张量重塑为(c*h*w)块作为特征向量
#     b, s, c, h, w = tensor.shape
#     tensor_reshaped = tensor.reshape(b * s, c * h * w)
#
#     # 初始化GPU资源
#     res = faiss.StandardGpuResources()
#
#     # 使用FAISS的GPU版本进行聚类
#     n_data, dim = tensor_reshaped.shape
#     flat_config = faiss.GpuIndexFlatConfig()
#     flat_config.device = 0  # 使用第一个GPU
#     kmeans = faiss.GpuIndexFlatL2(res, dim, flat_config)
#
#     kmeans = faiss.Kmeans(dim, n_clusters, niter=20, verbose=False, gpu=True)
#     kmeans.train(tensor_reshaped)
#     centroids = kmeans.centroids
#     _, assignments = kmeans.index.search(tensor_reshaped, 1)

    # return centroids, assignments
# def get_mask(features,clsnum):
#     n,c,h,w = features.shape
#     masks = []
#     mask_idxs = []
#     for i in range(n):
#         x = features[i].detach().cpu().numpy()
#         x = x.transpose(1,2,0)
#         x = x.reshape(-1,c)
#
#         # foreground/background cluster
#         _x = np.linalg.norm(x,axis=1,keepdims=True) #计算每行的L2范数并保留维度, l2范数就是平方和开根号
#         km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(_x)
#         bg_mask = km.labels_
#         ctrs = km.cluster_centers_
#         if ctrs[0][0] > ctrs[1][0]:
#             bg_mask = 1 - bg_mask
#         idx = np.where(bg_mask==1)[0]
#         if len(idx) <= 0.5*w*h:
#             continue
#         mask_idxs.append(i)
#
#         # pixel cluster
#         _x = x[idx]
#         cluster = KMeans(n_clusters=clsnum, n_init=10, random_state=0).fit(_x)
#         _res = cluster.labels_
#         res = np.zeros(h*w)
#         res[idx] = _res + 1
#
#         # align
#         res = res.reshape(h,w)
#         ys = []
#         for k in range(1,clsnum+1):
#             y = np.where(res==k)[0].mean()
#             ys.append(y)
#         ys = np.hstack(ys)
#         y_idxs = np.argsort(ys) + 1
#         heatmap = np.zeros_like(res)
#         for k in range(1,clsnum+1):
#             heatmap[res==y_idxs[k-1]] = k
#         masks.append(heatmap)
#     masks = np.stack(masks) if len(mask_idxs) > 0 else np.zeros(0)
#     mask_idxs = np.hstack(mask_idxs) if len(mask_idxs) > 0 else np.zeros(0)
#     return masks, mask_idxs


import torch
from kmeans_pytorch import kmeans


def get_mask(features, clsnum):
    device = features.device
    n, c, h, w = features.shape
    masks = []
    mask_idxs = []

    for i in range(n):
        x = features[i]
        #x = x.permute(1, 2, 0)
        x = x.reshape(-1, c)

        # foreground/background cluster
        _x = torch.norm(x, p=2, dim=1, keepdim=True)

        cluster_ids_x, cluster_centers = kmeans(X=_x, num_clusters=2, distance='euclidean', device=device)

        if cluster_centers[0] > cluster_centers[1]:
            cluster_ids_x = 1 - cluster_ids_x

        bg_mask = (cluster_ids_x == 0).nonzero().squeeze().to(device)

        if bg_mask.numel() <= 0.5 * w * h:
            continue
        mask_idxs.append(i)

        # pixel cluster
        _x = x[bg_mask]
        cluster_ids_x, _ = kmeans(X=_x, num_clusters=clsnum, distance='euclidean', device=device)
        _res = cluster_ids_x.to(device)
        res = torch.zeros(h * w, dtype=torch.long, device=device)
        res[bg_mask] = _res + 1

        # align
        res = res.reshape(h, w)
        ys = []
        for k in range(1, clsnum + 1):
            y = (res == k).nonzero(as_tuple=True)[0].float().mean()
            ys.append(y)
        ys = torch.stack(ys)
        y_idxs = torch.argsort(ys) + 1
        heatmap = torch.zeros_like(res)
        for k in range(1, clsnum + 1):
            heatmap[res == y_idxs[k - 1]] = k
        masks.append(heatmap)

    masks = torch.stack(masks) if len(mask_idxs) > 0 else torch.zeros(0, device=device)
    mask_idxs = torch.tensor(mask_idxs, device=device) if len(mask_idxs) > 0 else torch.zeros(0, device=device)

    return masks, mask_idxs


def get_mask_GPU(features,clsnum):
    n,c,h,w = features.shape
    masks = []
    mask_idxs = []
    for i in range(n):
        # x = features[i].detach().cpu().numpy()
        # x = x.transpose(1,2,0)
        # x = x.reshape(-1,c)
        x = x.view(-1,c)

        # foreground/background cluster
        #_x = np.linalg.norm(x,axis=1,keepdims=True) #计算每行的L2范数并保留维度, l2范数就是平方和开根号
        _x = torch.norm(x,p=2,dim=1,keepdim=True)

        km = kmeans(_x,2, distance='euclidean', device=torch.device('cuda:0'))

        #km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(_x)
        bg_mask = km[0]
        ctrs = km[-1]
        if ctrs[0][0] > ctrs[1][0]:
            bg_mask = 1 - bg_mask
        idx = np.where(bg_mask==1)[0]
        if len(idx) <= 0.5*w*h:
            continue
        mask_idxs.append(i)

        # pixel cluster
        _x = x[idx]
        #cluster = KMeans(n_clusters=clsnum, n_init=10, random_state=0).fit(_x)
        cluster = kmeans(_x, clsnum, distance='euclidean', device=torch.device('cuda:0'))

        _res = cluster[0]
        res = np.zeros(h*w)
        res[idx] = _res + 1

        # align
        res = res.reshape(h,w)
        ys = []
        for k in range(1,clsnum+1):
            y = np.where(res==k)[0].mean()
            ys.append(y)
        ys = np.hstack(ys)
        y_idxs = np.argsort(ys) + 1
        heatmap = np.zeros_like(res)
        for k in range(1,clsnum+1):
            heatmap[res==y_idxs[k-1]] = k
        masks.append(heatmap)
    masks = np.stack(masks) if len(mask_idxs) > 0 else np.zeros(0)
    mask_idxs = np.hstack(mask_idxs) if len(mask_idxs) > 0 else np.zeros(0)
    return masks, mask_idxs


def get_mask_v2(features, clsnum):
    b, s, c, h, w = features.shape
    all_masks = []
    all_mask_idxs = []

    for i in range(b):
        for j in range(s):
            x = features[i][j].detach().cpu().numpy()
            x = x.transpose(1, 2, 0)
            x = x.reshape(-1, c)

            # foreground/background cluster
            _x = np.linalg.norm(x, axis=1, keepdims=True)  # 计算每行的L2范数并保留维度
            km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(_x)
            bg_mask = km.labels_
            ctrs = km.cluster_centers_
            if ctrs[0][0] > ctrs[1][0]:
                bg_mask = 1 - bg_mask
            idx = np.where(bg_mask == 1)[0]
            if len(idx) <= 0.5 * w * h:
                continue
            all_mask_idxs.append((i, j))

            # pixel cluster
            _x = x[idx]
            cluster = KMeans(n_clusters=clsnum, n_init=10, random_state=0).fit(_x)
            _res = cluster.labels_
            res = np.zeros(h * w)
            res[idx] = _res + 1

            # align
            res = res.reshape(h, w)
            ys = []
            for k in range(1, clsnum + 1):
                y = np.where(res == k)[0].mean()
                ys.append(y)
            ys = np.hstack(ys)
            y_idxs = np.argsort(ys) + 1
            heatmap = np.zeros_like(res)
            for k in range(1, clsnum + 1):
                heatmap[res == y_idxs[k - 1]] = k
            all_masks.append(heatmap)

    all_masks = np.stack(all_masks) if len(all_mask_idxs) > 0 else np.zeros((0, h, w))
    return all_masks, all_mask_idxs

def extract_tensor_based_on_values(aa, cc):
    """
    根据给定的值从张量aa中提取相应的部分。

    参数:
        aa (torch.Tensor): 输入张量，形状为 [seq, 1024, 12, 4]
        cc (torch.Tensor): mask张量，形状为 [seq, 12, 4]

    返回:
        three tensors: 对应于cc中值1、2、3的提取结果，每个形状都是 [1024, n]，其中n是cc中对应值的数量
    """

    def extract_value(value, aa, cc):
        temp_results = []
        for aa_batch, cc_batch in zip(aa, cc):
            indices = torch.where(cc_batch == value)
            # 获取符合条件的张量切片
            masked_data = aa_batch[:, indices[0], indices[1]]
            temp_results.append(masked_data.reshape(-1, masked_data.shape[-1]))

        all_results = torch.cat(temp_results, dim=1)
        return all_results

    result1 = extract_value(1, aa, cc)
    result2 = extract_value(2, aa, cc)
    result3 = extract_value(3, aa, cc)

    return result1, result2, result3


def swap_patches(tensor, num_patches=2):
    """
    Swap random patches within the second dimension (of size 4) of the tensor.

    Parameters:
    - tensor: the input tensor of shape (16, 4, 1024, 12, 4)
    - num_patches: number of patches to be swapped

    Returns:
    - tensor with swapped patches
    """
    # Assuming the input tensor shape is (16, 4, 1024, 12, 4)
    batch_size, t, c, h, w = tensor.shape

    # We'll work with each item in the batch separately
    for i in range(batch_size):
        # Randomly choose "num_patches" from the (12, 4) dimension
        patch_indices = torch.randperm(h * w)[:num_patches].tolist()

        # Get the (h, w) indices for the chosen patches
        patch_coords = [(idx // w, idx % w) for idx in patch_indices]

        # Randomly shuffle the second dimension (of size 4)
        t_permutation = torch.randperm(t)

        # Perform the swapping
        for x, y in patch_coords:
            tensor[i, :, :, x, y] = tensor[i, t_permutation, :, x, y]

    return tensor
def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)


    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.dropout_rate > 0:
            feat = self.dropout(feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            elif 'module' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    #  def load_param(self, trained_path):
        #  param_dict = torch.load(trained_path, map_location = 'cpu')
        #  for i in param_dict:
            #  try:
                #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            #  except:
                #  continue
        #  print('Loading pretrained model from {}'.format(trained_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        convert_weights = True if pretrain_choice == 'imagenet' else False
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, pretrained=model_path, convert_weights=convert_weights, semantic_weight=semantic_weight)
        if model_path != '':
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        #if pretrain_choice == 'self':
        #    self.load_param(model_path)
    # input : x tensor bs,3,h,w | label tensor bs  cam_label tensor bs, view_label tensor bs   . x是一个batch的 img label 是personid ， camlabel是camid viewlabel是viewid，在market1501中，camid是1-6，viewid是1-6，personid都是1
    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat, featmaps = self.base(x)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)
            # output cls_score tensor bs pid_num(625) , global_feat bs 1024, featmaps list 4 features 128 96 32 ,  256 48 16, 512 24 8, 1024 12 4
            return cls_score, global_feat, featmaps  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat, featmaps
            else:
                # print("Test with feature before BN")
                return global_feat, featmaps

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))

class build_mars_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super(build_mars_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA: #使用  swin就没法用 camera信息
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        convert_weights = True if pretrain_choice == 'imagenet' else False
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, pretrained=model_path, convert_weights=convert_weights, semantic_weight=semantic_weight)
        if model_path != '':
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.attention_pooling = SelfAttentionPooling(self.in_planes)

        # -------------------video attention-------------
        self.middle_dim = 256  # middle layer dimension
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1, 1])  # 7,4 cooresponds to 224, 112 input image size  Conv2d(768, 256, kernel_size=[1, 1], stride=(1, 1))
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming)
        self.attention_tconv.apply(weights_init_kaiming)


        ###加入  a_val


        #if pretrain_choice == 'self':
        #    self.load_param(model_path)
    # input : x tensor bs,3,h,w | label tensor bs  cam_label tensor bs, view_label tensor bs   . x是一个batch的 img label 是personid ， camlabel是camid viewlabel是viewid，在market1501中，camid是1-6，viewid是1-6，personid都是1
    def forward(self, x, label=None, cam_label= None, view_label=None,clusting_feature=True,temporal_attention=False):
        model_jpm = False
        #clusting_feature = True

        b=x.size(0) # batch size 32
        t=x.size(1) # seq 4
        x = x.view(b * t, x.size(2), x.size(3), x.size(4)) #[32,4,3,256,128] --> [128,3,256,128]

        video = False
        if video:
            global_feat, featmaps = self.base(x,batchsize=b,seq_len=t,video=video) #如果是swin global 对应这 vid里的feat 也就是再来个 classifier就到score了
        else:

            global_feat, featmaps = self.base(x) #如果是swin global 对应这 vid里的feat 也就是再来个 classifier就到score了

        # ####随便混合一下
        # featmap_last = featmaps[-1]
        # featmap_last = featmap_last.view(b, t, featmap_last.size(1), featmap_last.size(2), featmap_last.size(3))
        #
        # featmap_last = swap_patches(featmap_last, num_patches=random.randint(0, 48))
        #
        # featmap_last = self.avgpool(featmap_last[-1]) # x就是 featue的最后一位的avgpool feature是64 1024 12 4
        # featmap_last = torch.flatten(featmap_last, 1)
        # global_feat = featmap_last


        #featmap_last = featmaps[-1]
        #featmap_last = featmap_last.view(b, t, featmap_last.size(1), featmap_last.size(2), featmap_last.size(3))
        #aaa = cluster_tensor_on_gpu(featmap_last, 3)

        if not video:
            #temporal attention
            if temporal_attention:
                global_feat = global_feat.unsqueeze(dim=2)
                global_feat = global_feat.unsqueeze(dim=3)
                a = F.relu(self.attention_conv(global_feat))
                a = a.view(b,t, self.middle_dim)
                a = a.permute(0,2,1)
                a = F.relu(self.attention_tconv(a))
                a = a.view(b,-1)
                a_val = a
                a = F.softmax(a, dim=1)
                x = global_feat.view(b,t,-1)
                a = torch.unsqueeze(a,-1)
                a = a.expand(b,t,self.in_planes)
                att_x = torch.mul(x,a)
                att_x = torch.sum(att_x, dim=1)
                global_feat = att_x.view(b,-1)

            else:
                global_feat = torch.mean(global_feat.view(-1, t, 1024), dim=1)

        if clusting_feature:
            featmap_last = featmaps[-1]
            featmap_last = featmap_last.view(b,t,featmap_last.size(1),featmap_last.size(2),featmap_last.size(3))
            part1_features = []
            part2_features = []
            part3_features = []
            for i in range(b):
                featmap_single = featmap_last[i]
                # mask = get_mask(featmap_single, 3)
                # mask_index = torch.from_numpy(mask[-1]).to(featmap_last.device)
                mask, mask_index = get_mask(featmap_single, 3)



                if mask_index.numel() != 0:
                    featmap_single = featmap_single[mask_index]
                    #mask_tensor = torch.from_numpy(mask[0]).to(featmap_last.device)
                    mask_tensor = mask
                    part1_feature = extract_tensor_based_on_values(featmap_single, mask_tensor)[0]
                    part2_feature = extract_tensor_based_on_values(featmap_single, mask_tensor)[1]
                    part3_feature = extract_tensor_based_on_values(featmap_single, mask_tensor)[2]

                    part1_feature = torch.mean(part1_feature,dim=1)
                    part2_feature = torch.mean(part2_feature,dim=1)
                    part3_feature = torch.mean(part3_feature,dim=1)
                else:
                    part1_feature = global_feat[i]
                    part2_feature = global_feat[i]
                    part3_feature = global_feat[i]

                part1_features.append(part1_feature)
                part2_features.append(part2_feature)
                part3_features.append(part3_feature)



            part1_features = torch.stack(part1_features)
            part2_features = torch.stack(part2_features)
            part3_features = torch.stack(part3_features)





        #变回bs * dim的形式，新加入
        # global_feat = torch.mean(global_feat.view(-1,t,1024),dim=1)

        #self-attention pooling
        #global_feat = self.attention_pooling(global_feat.view(-1,t,1024))

        if not model_jpm:
            if self.reduce_feat_dim:
                global_feat = self.fcneck(global_feat)
            feat = self.bottleneck(global_feat)
            feat_cls = self.dropout(feat)
            if clusting_feature:
                part1_feat = self.bottleneck(part1_features)
                part2_feat = self.bottleneck(part2_features)
                part3_feat = self.bottleneck(part3_features)
                part1_feat_cls = self.dropout(part1_feat)
                part2_feat_cls = self.dropout(part2_feat)
                part3_feat_cls = self.dropout(part3_feat)

            if self.training:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat_cls, label)
                else:
                    if clusting_feature:
                        cls_score = self.classifier(feat_cls)
                        cls_score_part1 = self.classifier(part1_feat_cls)
                        cls_score_part2 = self.classifier(part2_feat_cls)
                        cls_score_part3 = self.classifier(part3_feat_cls)
                        return [cls_score,cls_score_part1,cls_score_part2,cls_score_part3], [global_feat,part1_features,part2_features,part3_features], featmaps
                    else:
                        if not temporal_attention:
                            cls_score = self.classifier(feat_cls)
                            return cls_score, global_feat, featmaps
                        else:
                            cls_score = self.classifier(feat_cls)
                            return cls_score, global_feat, featmaps, a_val

                # output cls_score tensor bs pid_num(625) , global_feat bs 1024, featmaps list 4 features 128 96 32 ,  256 48 16, 512 24 8, 1024 12 4
                #return cls_score, global_feat, featmaps  # global feature for triplet loss,输出位
            else:
                if self.neck_feat == 'after':
                    #print("Test with feature after BN")
                    return feat, featmaps
                else:
                    #print("Test with feature before BN")
                    if clusting_feature:
                        return global_feat + (part1_features + part2_features + part3_features) / 3 , featmaps
                    else:
                        return global_feat, featmaps

                    #return global_feat, featmaps #输出位
        # else:




        # if self.reduce_feat_dim:
        #     global_feat = self.fcneck(global_feat)
        # feat = self.bottleneck(global_feat)
        # feat_cls = self.dropout(feat)
        #
        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat_cls, label)
        #     else:
        #         cls_score = self.classifier(feat_cls)
        #     # output cls_score tensor bs pid_num(625) , global_feat bs 1024, featmaps list 4 features 128 96 32 ,  256 48 16, 512 24 8, 1024 12 4
        #     return cls_score, global_feat, featmaps  # global feature for triplet loss
        # else:
        #     if self.neck_feat == 'after':
        #         print("Test with feature after BN")
        #         return feat, featmaps
        #     else:
        #         print("Test with feature before BN")
        #         return global_feat, featmaps

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))



class build_marspose_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super(build_marspose_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA: #使用  swin就没法用 camera信息
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        convert_weights = True if pretrain_choice == 'imagenet' else False
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, pretrained=model_path, convert_weights=convert_weights, semantic_weight=semantic_weight)
        if model_path != '':
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

            self.classifier_vit = nn.Linear(768, self.num_classes, bias=False)
            self.classifier_vit.apply(weights_init_classifier)


        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_vit = nn.BatchNorm1d(768)
        self.bottleneck_vit.bias.requires_grad_(False)
        self.bottleneck_vit.apply(weights_init_kaiming)


        self.dropout = nn.Dropout(self.dropout_rate)

        self.attention_pooling = SelfAttentionPooling(self.in_planes)


        ###加入  a_val


        #if pretrain_choice == 'self':
        #    self.load_param(model_path)
    # input : x tensor bs,3,h,w | label tensor bs  cam_label tensor bs, view_label tensor bs   . x是一个batch的 img label 是personid ， camlabel是camid viewlabel是viewid，在market1501中，camid是1-6，viewid是1-6，personid都是1
    def forward(self, x,keypoints, label=None, cam_label= None, view_label=None):
        model_jpm = False

        b=x.size(0) # batch size 32
        t=x.size(1) # seq 4
        x = x.view(b * t, x.size(2), x.size(3), x.size(4)) #[32,4,3,256,128] --> [128,3,256,128]
        #keypoints = keypoints.view(b * t, keypoints.size(2), keypoints.size(3))

        if self.training:
            global_feat, featmaps,keypointsfeature = self.base(x,keypoints,b,t) #如果是swin global 对应这 vid里的feat 也就是再来个 classifier就到score了
        else:
            global_feat, featmaps = self.base(x)




        #变回bs * dim的形式，新加入
        global_feat = torch.mean(global_feat.view(-1,t,1024),dim=1)

        #self-attention pooling
        #global_feat = self.attention_pooling(global_feat.view(-1,t,1024))

        if not model_jpm:
            if self.reduce_feat_dim:
                global_feat = self.fcneck(global_feat)
            feat = self.bottleneck(global_feat)
            #vit_feat = self.boottleneck_vit(keypointsfeature)
            feat_cls = self.dropout(feat)
            if self.training:
                keypointsfeature_cls = self.dropout(keypointsfeature)

            if self.training:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat_cls, label)
                else:
                    cls_score = self.classifier(feat_cls)
                    cls_score_vit = self.classifier_vit(keypointsfeature_cls)
                return cls_score, global_feat, featmaps,cls_score_vit,keypointsfeature  # global feature for triplet loss,输出位
            else:
                if self.neck_feat == 'after':
                    #print("Test with feature after BN")
                    return feat, featmaps
                else:
                    #print("Test with feature before BN")
                    return global_feat, featmaps #输出位
        # else:




        # if self.reduce_feat_dim:
        #     global_feat = self.fcneck(global_feat)
        # feat = self.bottleneck(global_feat)
        # feat_cls = self.dropout(feat)
        #
        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat_cls, label)
        #     else:
        #         cls_score = self.classifier(feat_cls)
        #     # output cls_score tensor bs pid_num(625) , global_feat bs 1024, featmaps list 4 features 128 96 32 ,  256 48 16, 512 24 8, 1024 12 4
        #     return cls_score, global_feat, featmaps  # global feature for triplet loss
        # else:
        #     if self.neck_feat == 'after':
        #         print("Test with feature after BN")
        #         return feat, featmaps
        #     else:
        #         print("Test with feature before BN")
        #         return global_feat, featmaps

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))

# class build_mars_transformer(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
#         super(build_mars_transformer, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.in_planes = self.base.in_planes
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         block = self.base.blocks[-1]
#         layer_norm = self.base.norm
#         self.b1 = nn.Sequential(
#             copy.deepcopy(block),
#             copy.deepcopy(layer_norm)
#         )
#         self.b2 = nn.Sequential(
#             copy.deepcopy(block),
#             copy.deepcopy(layer_norm)
#         )
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             self.classifier.apply(weights_init_classifier)
#             self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             self.classifier_1.apply(weights_init_classifier)
#             self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             self.classifier_2.apply(weights_init_classifier)
#             self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             self.classifier_3.apply(weights_init_classifier)
#             self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             self.classifier_4.apply(weights_init_classifier)
#
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)
#         self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_1.bias.requires_grad_(False)
#         self.bottleneck_1.apply(weights_init_kaiming)
#         self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_2.bias.requires_grad_(False)
#         self.bottleneck_2.apply(weights_init_kaiming)
#         self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_3.bias.requires_grad_(False)
#         self.bottleneck_3.apply(weights_init_kaiming)
#         self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_4.bias.requires_grad_(False)
#         self.bottleneck_4.apply(weights_init_kaiming)
#
#         self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
#         print('using shuffle_groups size:{}'.format(self.shuffle_groups))
#         self.shift_num = cfg.MODEL.SHIFT_NUM
#         print('using shift_num size:{}'.format(self.shift_num))
#         self.divide_length = cfg.MODEL.DEVIDE_LENGTH
#         print('using divide_length size:{}'.format(self.divide_length))
#         self.rearrange = rearrange
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         b = x.size(0)  # batch size 32
#         t = x.size(1)  # seq 4
#         x = x.view(b * t, x.size(2), x.size(3), x.size(4))  # [32,4,3,256,128] --> [128,3,256,128]
#
#         features = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global branch
#         b1_feat = self.b1(features) # [64, 129, 768]
#         global_feat = b1_feat[:, 0]
#
#         # JPM branch
#         feature_length = features.size(1) - 1
#         patch_length = feature_length // self.divide_length
#         token = features[:, 0:1]
#
#         if self.rearrange:
#             x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
#         else:
#             x = features[:, 1:]
#         # lf_1
#         b1_local_feat = x[:, :patch_length]
#         b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
#         local_feat_1 = b1_local_feat[:, 0]
#
#         # lf_2
#         b2_local_feat = x[:, patch_length:patch_length*2]
#         b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
#         local_feat_2 = b2_local_feat[:, 0]
#
#         # lf_3
#         b3_local_feat = x[:, patch_length*2:patch_length*3]
#         b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
#         local_feat_3 = b3_local_feat[:, 0]
#
#         # lf_4
#         b4_local_feat = x[:, patch_length*3:patch_length*4]
#         b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
#         local_feat_4 = b4_local_feat[:, 0]
#
#         feat = self.bottleneck(global_feat)
#
#         local_feat_1_bn = self.bottleneck_1(local_feat_1)
#         local_feat_2_bn = self.bottleneck_2(local_feat_2)
#         local_feat_3_bn = self.bottleneck_3(local_feat_3)
#         local_feat_4_bn = self.bottleneck_4(local_feat_4)
#
#         if self.training:
#             if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
#                 cls_score = self.classifier(feat, label)
#             else:
#                 cls_score = self.classifier(feat)
#                 cls_score_1 = self.classifier_1(local_feat_1_bn)
#                 cls_score_2 = self.classifier_2(local_feat_2_bn)
#                 cls_score_3 = self.classifier_3(local_feat_3_bn)
#                 cls_score_4 = self.classifier_4(local_feat_4_bn)
#             return [cls_score, cls_score_1, cls_score_2, cls_score_3,
#                         cls_score_4
#                         ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
#                             local_feat_4]  # global feature for triplet loss
#         else:
#             if self.neck_feat == 'after':
#                 return torch.cat(
#                     [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
#             else:
#                 return torch.cat(
#                     [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))



class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))



__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
    'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
    'pose_swin_base_patch4_window12_384': pose_swin_base_patch4_window7_224,
}

def make_model(cfg, num_class, camera_num, view_num, semantic_weight):
    if cfg.MODEL.NAME == 'transformer':

        if cfg.DATASETS.NAMES == 'mars':
            model = build_mars_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
            print('===========building mars transformer===========')

        elif cfg.DATASETS.NAMES == 'marspose':
            model = build_marspose_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
            print('===========building marspose transformer===========')
        else:
            if cfg.MODEL.JPM:
                model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type,
                                                rearrange=cfg.MODEL.RE_ARRANGE)
                print('===========building transformer with JPM module ===========')
            else:
                model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
                print('===========building transformer===========')

        # if cfg.MODEL.JPM:
        #     model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
        #     print('===========building transformer with JPM module ===========')
        # else:
        #     model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
        #     print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model

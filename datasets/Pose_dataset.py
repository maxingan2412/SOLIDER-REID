from __future__ import division, print_function, absolute_import
import pickle
#from torchreid.data.datasets import VideoDataset
#from torchreid.utils import read_image

#from ...utils.tools import read_keypoints,read_keypoints_mars

import copy
import numpy as np
import os.path as osp
import tarfile
import zipfile
import torch
import os
import errno

#from torchreid.utils import read_image, download_url, mkdir_if_missing
from PIL import Image

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_keypoints_posetrackreid(path):
    pkl_root = '/home/ma1/work/data/posetrack-reid/keypoints_cache.pkl'
    with open(pkl_root, 'rb') as f:
        # with open(server, 'rb') as f:
        keypoints_cache = pickle.load(f)


    file_name = path.split('/')[-1]
    kps = keypoints_cache[file_name]  #这里load了 xxx.pose  读取之后 kps就是 一个 ndarry  17*3的
    return kps.copy()

def read_keypoints_mars(path):
    """Reads keypoints from path.
    Args:
        path (str): path to a keypoints file.
    Returns:
        numpy array
    """
    got_kp = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_kp:
        try:
            with open(path, 'rb')as f:
                kps = f.readline()  # one-line only text
            kps = np.asarray(eval(kps), dtype=np.float32)
            kps[kps != kps] = 0.0  # detect nan and replace nan with 0
            got_kp = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return kps

def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """

    # junk_pids contains useless person IDs, e.g. background,
    # false detections, distractors. These IDs will be ignored
    # when combining all images in a dataset for training, i.e.
    # combineall=True
    _junk_pids = []

    # Some datasets are only used for training, like CUHK-SYSU
    # In this case, "combineall=True" is not used for them
    _train_only = False

    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        k_tfm=1,
        mode='train',
        combineall=False,
        verbose=True,
        **kwargs
    ):
        # extend 3-tuple (img_path(s), pid, camid) to
        # 4-tuple (img_path(s), pid, camid, dsetid) by
        # adding a dataset indicator "dsetid"
        if len(train[0]) == 3:
            train = [(*items, 0) for items in train] #把后面加个0，表示dsetid
        if len(query[0]) == 3:
            query = [(*items, 0) for items in query]
        if len(gallery[0]) == 3:
            gallery = [(*items, 0) for items in gallery]

        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.k_tfm = k_tfm
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        self.num_datasets = self.get_num_datasets(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | query | gallery]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for img_path, pid, camid, dsetid in other.train:
            pid += self.num_train_pids
            camid += self.num_train_cams
            dsetid += self.num_datasets
            train.append((img_path, pid, camid, dsetid))

        ###################################
        # Note that
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset; setting it to True will
        #    create new IDs that should have already been included
        ###################################
        if isinstance(train[0][0], str):
            return ImageDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False
            )
        else:
            return VideoDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
                seq_len=self.seq_len,
                sample_method=self.sample_method
            )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_num_pids(self, data):
        """Returns the number of training person identities.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        pids = set()
        for items in data:
            pid = items[1]
            pids.add(pid)
        return len(pids)

    def get_num_cams(self, data):
        """Returns the number of training cameras.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        cams = set()
        for items in data:
            camid = items[2]
            cams.add(camid)
        return len(cams)

    def get_num_datasets(self, data):
        """Returns the number of datasets included.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        dsets = set()
        for items in data:
            dsetid = items[3]
            dsets.add(dsetid)
        return len(dsets)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        if self._train_only:
            return

        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for items in self.gallery:
            pid = items[1]
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid, dsetid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid, dsetid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        #download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
                  num_train_pids, len(self.train), num_train_cams,
                  num_query_pids, len(self.query), num_query_cams,
                  num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg

    def _transform_image(self, tfm, k_tfm, img0):
        """Transforms a raw image (img0) k_tfm times with
        the transform function tfm.
        """
        img_list = []

        for k in range(k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid, dsetid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }
        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')


class VideoDataset(Dataset):
    """A base class representing VideoDataset.

    All other video datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``imgs``, ``pid`` and ``camid``
    where ``imgs`` has shape (seq_len, channel, height, width). As a result,
    data in each batch has shape (batch_size, seq_len, channel, height, width).
    """

    def __init__(
        self,
        train,
        query,
        gallery,
        seq_len=15,
        sample_method='evenly',
        **kwargs
    ):
        super(VideoDataset, self).__init__(train, query, gallery, **kwargs)
        self.seq_len = seq_len
        self.sample_method = sample_method

        if self.transform is None:
            raise RuntimeError('transform must not be None')

    def __getitem__(self, index):
        img_paths, pid, camid, dsetid = self.data[index]
        num_imgs = len(img_paths)

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= self.seq_len else True
            indices = np.random.choice(
                indices, size=self.seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= self.seq_len:
                num_imgs -= num_imgs % self.seq_len
                indices = np.arange(0, num_imgs, num_imgs / self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = self.seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs-1)
                    ]
                )
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0) # img must be torch.Tensor
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        #mar20 可能这里不一样所以video的数据和pose 的不一样
        item = {'img': imgs, 'pid': pid, 'camid': camid, 'dsetid': dsetid}   # pcb 的posetrackreidvo 指向这里。 item含有的 就是pcb中的input含有的。

        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)
        print(' father class is Dataset,which have self.show_summary() that is override in videoDataset, so this is videoDataset show_summary()')
        print('=> here is Class VideoDataset ------ Loaded {}'.format(self.__class__.__name__))
        print('  -------------------------------------------')
        print('  subset   | # ids | # tracklets | # cameras')
        print('  -------------------------------------------')
        print(
            '  train    | {:5d} | {:11d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:11d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:11d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  -------------------------------------------')




'''
class Joints:              class JointsHeadless:
    Nose = 0
    Leye = 1
    Reye = 2
    LEar = 3
    REar = 4
    LS = 5                     LS = 0
    RS = 6                     RS = 1
    LE = 7                     LE = 2
    RE = 8                     RE = 3
    LW = 9                     LW = 4
    RW = 10                    RW = 5
    LH = 11                    LH = 6
    RH = 12                    RH = 7
    LK = 13                    LK = 8
    RK = 14                    RK = 9
    LA = 15                    LA = 10
    RA = 16                    RA = 11

'''
#意思是哪些关节是相连的
link_pairs = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6],
    [6, 8], [8, 10],
    # reverse connection
    [13, 15], [11, 13], [5, 11],
    [14, 12], [16, 14], [6, 12],
    [1, 3], [2, 1], [0, 1], [2, 0], [4, 2],
    [7, 9], [5, 7], [6, 5],
    [8, 6], [10, 8]
]


link_pairs_headless = [
    [10, 8], [8, 6], [6, 0],
    [7, 9], [9, 11], [7, 1],
    [4, 2], [2, 0], [0, 1],
    [1, 3], [3, 5],
    # reverse connection
    [8, 10], [6, 8], [0, 6],
    [9, 7], [11, 9], [1, 7],
    [2, 4], [0, 2], [1, 0],
    [3, 1], [5, 3]
]


oks_sigmas = torch.Tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                           1.07, 1.07, .87, .87, .89, .89]) / 10.0


def normalize_pose(pf, keep_orig_coord=False,
                   orig_img_width=None, orig_img_height=None, epsilon=2.0):
    if keep_orig_coord:
        orig_coord = pf.copy()[:, :2]
        orig_coord[:, 0] /= orig_img_width
        orig_coord[:, 1] /= orig_img_height

    lx, ly = np.min(pf[:, 0]), np.min(pf[:, 1])
    pf[:, 0] -= lx
    pf[:, 1] -= ly
    pose_width = np.max(pf[:, 0])
    pose_height = np.max(pf[:, 1])
    if pose_width == 0:
        pose_width = epsilon
    if pose_height == 0:
        pose_height = epsilon    
    ct_x = np.mean(pf[:, 0])
    ct_y = np.mean(pf[:, 1])
    pf[:, 0] = (pf[:, 0] - ct_x) / (pose_width / 2.0)
    pf[:, 1] = (pf[:, 1] - ct_y) / (pose_height / 2.0)
    if keep_orig_coord:
        return np.concatenate([orig_coord, pf], axis=1)
    else:
        return pf


def select_window(index, length, window_size=5):
    if index <= window_size // 2:
        return list(range(window_size))
    elif index >= length - window_size // 2:
        return list(range(length - window_size, length))
    else:
        return list(range(index - window_size // 2, index + 1 + window_size // 2))


def oks_iou(pose_i, pose_j, scale=1.0, sigmas=oks_sigmas):
    '''
    pose_i: torch.Tensor, size = (17 or 12, 3 or 5)
    pose_j: same to pose_i
    '''
    if pose_i.shape[0] == 12:
        sigmas = sigmas[5:]

    if pose_i.shape[1] == 5:
        dx = pose_i[:, 2] - pose_j[:, 2]  # do it in the normalized pose space
        dy = pose_i[:, 3] - pose_j[:, 3]
    elif pose_i.shape[1] == 3:
        dx = pose_i[:, 0] - pose_j[:, 0]
        dy = pose_i[:, 1] - pose_j[:, 1]
    e = (dx**2 + dy**2) / (2 * scale**2 * sigmas**2)
    return torch.sum(torch.exp(-e)) / pose_i.shape[0]


def filter_glitches(keypoints, thresh=0.2, window_size=5):
    '''
    keypoints: torch.Tensor, size = (num_frames, num_keypoints, 3)
    '''
    num_frames = keypoints.shape[0]
    for i in range(num_frames):
        window_index = select_window(i, num_frames, window_size)
        
        s = []
        for j in window_index:
            if j != i:
                s.append(oks_iou(keypoints[i], keypoints[j]))

        if np.mean(s) < thresh:  # frame i is a glitch
            if i == 0:
                keypoints[i] = keypoints[i + 1]
            elif i == num_frames - 1:
                keypoints[i] = keypoints[i - 1]
            else:
                keypoints[i] = (keypoints[i - 1] * 0.8 + keypoints[i + 1] * 0.2)  
                # previous frames are more reliable since it's already fixed.
    return keypoints


def append_dynamics_features(keypoints, velocity=True, acceleration=True):
    '''
    keypoints: torch.Tensor, size = (num_frames, num_keypoints, 3 or 5)
    '''
    if keypoints.shape[2] == 3:
        position = keypoints[:, :, :2]  # size = (num_frames, num_keypoints, 2)
        confidence = keypoints[:, :, 2:]  # size = (num_frames, num_keypoints, 1)
    elif keypoints.shape[2] == 5:
        orig_position = keypoints[:, :, :2]
        position = keypoints[:, :, 2:4]  # size = (num_frames, num_keypoints, 2)
        confidence = keypoints[:, :, 4:]  # size = (num_frames, num_keypoints, 1)
    
    out = position

    if velocity:
        v = position[1:] - position[:-1]  # size = (num_frames-1, num_keypoints, 2)
        v = torch.cat([v, v[-1:]], dim=0)  # simply repeat the velocity of the last frame
        out = torch.cat([out, v], dim=2)

    if acceleration:
        a = v[1:] - v[:-1]
        a = torch.cat([a, a[-1:]], dim=0)
        out = torch.cat([out, a], dim=2)

    if keypoints.shape[2] == 3:
        out = torch.cat([out, confidence], dim=2)
    elif keypoints.shape[2] == 5:
        out = torch.cat([orig_position, out, confidence], dim=2)

    return out


class PoseDataset(VideoDataset):

    def __init__(
        self,
        train,
        query,
        gallery,
        seq_len=15,
        sample_method='evenly',
        return_img=False,
        return_pose_graph=True,
        headless=False,
        rm_glitches=False,
        include_dynamics=False,
        include_spatial_links=True,
        include_temporal_links=True,
        normalize_confidence=True,
        **kwargs
    ): # **kwargs 会把上面有的赋值过去，没有的就还存在这个字典里面
        super(PoseDataset, self).__init__(
            train, query, gallery,
            seq_len=seq_len, sample_method=sample_method, **kwargs)

        if self.transform is None:
            raise RuntimeError('transform must not be None')

        self.return_img = return_img
        self.return_pose_graph = return_pose_graph
        self.headless = headless
        self.rm_glitches = rm_glitches
        self.include_dynamics = include_dynamics
        self.include_spatial_links = include_spatial_links
        self.include_temporal_links = include_temporal_links
        self.keep_orig_coord = self.return_img
        self.normalize_confidence = normalize_confidence
    '''
    凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像这样 p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法__getitem__。 
    https://zhuanlan.zhihu.com/p/27661382
    
    这里也就是 输入一个idx 输出一个item 
    后面转成dataloader后，每一次迭代就会调用这个getiem方法获取单个样本，然后你dataloader制定了bs 所以他会根据这个bs组成一个batch，也就是说看这个getitem方法返回的是什么，过程中怎么处理的数据，就是看循环中得到了什么配合bs就是一个batch的数据
    '''
    def __getitem__(self, idx):
        # img_paths, pose_paths, pid, camid = self.data[idx]
        # num_imgs = len(img_paths)
        #print('daodiduochang',len(self.data[idx]))
        #-----error happened here ----
        global img_path
        pose_paths, pid, camid, _ = self.data[idx]
        num_imgs = len(pose_paths)
        sample_seq_len = self.seq_len + 2  # over-sample 2 imgs for velocity and acceleration

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= sample_seq_len else True #如果你图片多，就不用放回去，图片少，就放回去，因为不放回去就不够了
            '''
            # 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
            a1 = np.random.choice(a=5, size=3, replace=False, p=None)
             replace 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。
            '''

            indices = np.random.choice(
                indices, size=sample_seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be
            # order-agnostic)
            #np.sort就是把 array从小到大排列
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= sample_seq_len:
                num_imgs -= num_imgs % sample_seq_len
                indices = np.arange(0, num_imgs, num_imgs / sample_seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = sample_seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs - 1)
                    ]
                )
            assert len(indices) == sample_seq_len

        elif self.sample_method == 'conseq':
            # Samples concecutive seq_len images from a tracklet
            if num_imgs >= sample_seq_len:
                startpoint = np.random.randint(
                    low=0, high=num_imgs - sample_seq_len + 1)
                indices = np.arange(startpoint, startpoint + sample_seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = sample_seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs - 1)
                    ]
                )
            assert len(indices) == sample_seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs, keypoints = [], []
        img_paths = []
        for index in indices:
            keypoints_path = pose_paths[int(index)]
            if self.return_img:
                # img_path = img_paths[int(index)]
                img_path = keypoints_path.replace('.pose', '.jpg')
                img_path = img_path.replace('_keypoints', '')
                img = read_image(img_path) # single image
                orig_img_width, orig_img_height = img.size
                img_paths.append(img_path)
            else:
                img = None
                orig_img_width, orig_img_height = None, None

            #if self.dataset_dir == '/home/ma1/work/2021/PoseTrackReID/data/mars':
                #kps =
            #else:
            if self.dataset_dir.split('/')[-1] == 'mars':
                # with open(keypoints_path,'r') as f:
                #     loaded_list = eval(f.read())
                #     #kps = torch.tensor(loaded_list)
                #     kps = np.array(loaded_list)
                kps = read_keypoints_mars(keypoints_path)
            else:
                kps = read_keypoints_posetrackreid(keypoints_path)  # numpy array, size = (17, 3)
            if self.headless:
                kps = kps[5:]  # numpy array, size = (12, 3)
            # mar19
            if self.transform is not None:
                # img, kps = self.transform(img, kps)  # TODO!!!! re-write all
                # trandform functions
                if self.return_img:
                    img = self.transform(img)
                kps = normalize_pose(kps, self.keep_orig_coord, orig_img_width, orig_img_height) # 从17*3  --- 17*5了 orgin pf--也就是转换的kps 最后一位是置信度
                kps = torch.from_numpy(kps) # numpy zhuan torch
            if self.return_img:
                img = img.unsqueeze(0)  # img must be torch.Tensor 3 256 128 -- 1 3 256 128

            kps = kps.unsqueeze(0)  # torch.Tensor, size = (1, 17 or 12, 3 or 5)
            imgs.append(img)
            keypoints.append(kps)
        #上面生成的keypoints 的长度是根据上面的indice来的

        if self.return_img:
            imgs = torch.cat(imgs, dim=0) #list 变tensor，list长度是 sample_seq_len
        # torch.Tensor, size = (N, 17 or 12, 3)
        keypoints = torch.cat(keypoints, dim=0)
        if self.rm_glitches:
            keypoints = filter_glitches(keypoints)

        if self.include_dynamics: #这里可以添加 加速度和速度作为新的特征，但是速度是用相邻帧位置差计算，加速度是相邻速度差计算，这就要求这个list应该是有序的 ---似乎是有序的因为前面排序了一下，都是从前往后的。所以其实可以用。有可能后续有用
            keypoints = append_dynamics_features(keypoints)

        keypoints = keypoints[:self.seq_len]  # remove the over-sampled imgs  136*5 -- 8*17*5
        imgs = imgs[:self.seq_len]

        if self.return_pose_graph:
            keypoints, edge_index = self.graph_gen(keypoints, self.headless,
                                                   self.include_spatial_links,
                                                   self.include_temporal_links) # keypoinsts仅仅是变了形状，edge_index是利用关键点得到了链接方式

        if self.normalize_confidence:
            keypoints[:, -1] = keypoints[:, -1] - keypoints[:, -1].mean() + 1 # 置信度的均值将变成1


        #mar20 830
        item = {'img': imgs,
                'keypoints': keypoints,
                'edge_index': edge_index,
                'pid': pid,
                'camid': camid, #因为训练的时候就没用到camid 所以就随便给了一个值
                'img_path': img_paths}

        return item

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pose_path(s), pid, camid)
            update: data (list): contains tuples of (pose_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        # for _, _, pid, camid in data:
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def __add__(self, other):
        raise NotImplementedError

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        # for _, _, pid, _ in self.gallery:
        for _, pid, _ in self.gallery:
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            # for img_path, pose_path, pid, camid in data:
            for pose_path, pid, camid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                # combined.append((img_path, pose_path, pid, camid))
                combined.append((pose_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)
    #这里建立了keypoint的连接方式，好像是建立了一个seq_len中 keypoint 在图片内和图片间的关系 3.6evening
    @staticmethod
    def graph_gen(keypoints, headless=False, spatial=True, temporal=True):
        N, num_kps, dim = keypoints.shape

        if headless:
           base_pairs = link_pairs_headless
        else:
           base_pairs = link_pairs

        base_skelenton = torch.tensor(base_pairs, dtype=torch.long)

        edge_index = []
        if spatial:
            # intra-frame edges
            for i in range(N):
                edge_index.append(base_skelenton + i * num_kps) #其实就是创造 frame内部的链接方式，但是因为总共有N个所以他们的编号要不一样。但链接方式是一样的
        #
        if temporal:
            # inter_frame edges
            for i in range(N - 1):  # N - 1: the last frame has no connections to the next
                links = [[i * num_kps + j, (i + 1) * num_kps + j]
                         for j in range(num_kps)] # link在这里是12的list，也就是俩图片之间12个关键点的链接方式。
                edge_index.append(
                    torch.tensor(links, dtype=torch.long)
                )
                reverse_links = [[(i + 1) * num_kps + j, i * num_kps + j]
                                 for j in range(num_kps)] #可能是因为图网络有单项和双向的区别，所以这里拿到了反向的链接方式，也加上，因为上面spatial也是双向的
                edge_index.append(
                    torch.tensor(reverse_links, dtype=torch.long)
                )
        #edgeindex是一个list22 ---8个事tesnor22 2，表示的是图片内的关键点的链接方式， 22 -8=14 14 /2 = 7 描述的是 因为上面N=8，描述的是8个图片之间的链接关系，因为反向存在所以是14
        return keypoints.reshape(N * num_kps, dim), torch.cat(edge_index, dim=0)

from __future__ import division, print_function, absolute_import
import os.path as osp
import warnings
from scipy.io import loadmat
#报错的原因是
#from .Pose_dataset import PoseDataset
import numpy as np
from collections import defaultdict



# class MarsPoseold(PoseDataset):
#     """MARS.
#     Reference:
#         Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
#     URL: `<http://www.liangzheng.com.cn/Project/project_mars.html>`_
#
#     Dataset statistics:
#         - identities: 1261.
#         - tracklets: 8298 (train) + 1980 (query) + 9330 (gallery).
#         - keypoints: 17
#         - cameras: 6.
#     """
#     dataset_dir = 'mars'
#     dataset_url = None
#
#     def __init__(self, root='', **kwargs):
#         self.root = osp.abspath(osp.expanduser(root)) #'/home/ma1/work/PoseTrackReID'
#         self.dataset_dir = osp.join(self.root, self.dataset_dir) # '/home/ma1/work/PoseTrackReID/data/mars'
#         self.download_dataset(self.dataset_dir, self.dataset_url)
#
#         self.train_name_path = osp.join(
#             self.dataset_dir, 'info/train_name.txt'
#         )
#         self.test_name_path = osp.join(self.dataset_dir, 'info/test_name.txt')
#         self.track_train_info_path = osp.join(
#             self.dataset_dir, 'info/tracks_train_info.mat'
#         )
#         self.track_test_info_path = osp.join(
#             self.dataset_dir, 'info/tracks_test_info.mat'
#         )
#         self.query_IDX_path = osp.join(self.dataset_dir, 'info/query_IDX.mat')
#
#         required_files = [
#             self.dataset_dir, self.train_name_path, self.test_name_path,
#             self.track_train_info_path, self.track_test_info_path,
#             self.query_IDX_path
#         ]
#         self.check_before_run(required_files)
#
#         train_names = self.get_names(self.train_name_path)
#         test_names = self.get_names(self.test_name_path)
#         track_train = loadmat(self.track_train_info_path  #mat 是  8294 *4 的nndarray,  4维分别是     pid ，camid
#                               )['track_train_info']  # numpy.ndarray (8298, 4)
#         track_test = loadmat(self.track_test_info_path
#                              )['track_test_info']  # numpy.ndarray (12180, 4)
#         query_IDX = loadmat(self.query_IDX_path
#                             )['query_IDX'].squeeze()  # numpy.ndarray (1980,)
#         query_IDX -= 1  # index from 0
#         track_query = track_test[query_IDX, :] #按照idx找到 query的tracklets
#         gallery_IDX = [
#             i for i in range(track_test.shape[0]) if i not in query_IDX
#         ] #除了queryidx对应的tracklets，其他的都是gallery的tracklets
#         track_gallery = track_test[gallery_IDX, :]
#
#         train = self.process_data(  #list 8298 length(tracklets) .  each tracklet is list contains keypointspath pid camid
#             train_names, track_train, home_dir='bbox_train', relabel=True
#         ) #长度是一个8298的list 其中每个元素是一个tuple，tuple里面是一个tracklets的信息，一个包含该tracklet的图片对应的pose的地址的list，pid，camid
#         query = self.process_data(
#             test_names, track_query, home_dir='bbox_test', relabel=False
#         )
#         gallery = self.process_data(
#             test_names, track_gallery, home_dir='bbox_test', relabel=False
#         )
#
#         super(MarsPose, self).__init__(train, query, gallery, **kwargs)
#
#     def get_names(self, fpath):
#         names = []
#         with open(fpath, 'r') as f:
#             for line in f:
#                 new_line = line.rstrip()
#                 names.append(new_line)
#         return names
#
#     def process_data(
#         self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0
#     ):
#         assert home_dir in ['bbox_train', 'bbox_test']
#         num_tracklets = meta_data.shape[0]
#         pid_list = list(set(meta_data[:, 2].tolist()))  #pid 的个数，对于test 626
#
#         if relabel:
#             pid2label = {pid: label for label, pid in enumerate(pid_list)} #pid变成label，之前pid是数据集中的，是不连续的，pid2label是建立一个映射
#         tracklets = []
#
#         for tracklet_idx in range(num_tracklets):
#             data = meta_data[tracklet_idx, ...]
#             start_index, end_index, pid, camid = data   # 这四个也就是 mat中的4维
#             if pid == -1 or pid == 0:
#                 continue  # junk images are just ignored, '0000' and '00-1' not
#             # provided by team PoseTrack
#             assert 1 <= camid <= 6
#             if relabel:
#                 pid = pid2label[pid] #按照上面的映射，把pid变成label
#             camid -= 1  # index starts from 0
#             img_names = names[start_index - 1:end_index]
#
#             # make sure image names correspond to the same person
#             pnames = [img_name[:4] for img_name in img_names]
#             assert len(set(pnames)) == 1, \
#                 'Error: a single tracklet contains different person images'
#
#             # make sure all images are captured under the same camera
#             camnames = [img_name[5] for img_name in img_names]
#             assert len(set(camnames)) == 1, \
#                 'Error: images are captured under different cameras!'
#
#             # append image names with directory information
#             img_paths = [
#                 osp.join(self.dataset_dir, home_dir, img_name[:4], img_name) #这里在拼接路径，img_name[:4]比如是0001
#                 for img_name in img_names
#             ]
#             keypoints_paths = [
#                 osp.join(self.dataset_dir,
#                          home_dir + '_keypoints',
#                          img_name[:4],
#                          img_name.replace('.jpg', '.pose')) #找到对应的tracklets下的这些图片的keypoints
#                 for img_name in img_names
#             ]
#             if len(img_paths) >= min_seq_len: # 所以如果tracklets的长度小于min_seq_len，就舍弃了这个tracklets
#                 img_paths = tuple(img_paths)
#                 tracklets.append((keypoints_paths, pid, camid))
#
#         return tracklets
#
#     def combine_all(self):
#         warnings.warn(
#             'Some query IDs do not appear in gallery. Therefore, combineall '
#             'does not make any difference to Mars'
#         )

class MarsPose(object):

    # root  ='MARS' #'/home2/zwjx97/STE-NVAN-master/MARS' #"/home/aishahalsehaim/Desktop/STE-NVAN-master/MARS"
    root = '../data/MARS'

    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=0, ):
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)  # 这里得到的是一个列表，509914，每个形如 123131.jpg，是图片的名字
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)[
            'track_train_info']  # numpy.ndarray (8298, 4) 形如 [1, 16, 1, 1]， 前俩是图片序号，3是id，4是摄像头id
        track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)[
            'query_IDX'].squeeze()  # numpy.ndarray (1980,) .squeeze()：这是一个函数，用于删除数组中的单一维度。如果 query_IDX 是一个数组，可能有一个维度的大小为 1，使用 .squeeze() 可以将这个维度去除，使得数组更加紧凑。
        query_IDX -= 1  # index from 0 # query 并不是简单的1-1980，而是一堆很大的数 比如一万多4000多
        track_query = track_test[query_IDX, :]
        '''
        这个文件用matlab打开后可以看到是一个1 * 1980的矩阵，可以看到每一列是对应上面 tracks_test_info.mat文件中的第几行。
        比如1978列中的值为12177，对应的是 tracks_test_info.mat文件中的第12177行。这1980找到的tracklet的ID都是不一样的。

        '''

        # (1980, 4)
        gallery_IDX = [i for i in range(track_test.shape[0]) if
                       i not in query_IDX]  # 12180 - 1980 = 10200 个 是从0开始如果不是query的就是gallery，所以比如4129 gallery_IDX[4129] = 0
        track_gallery = track_test[gallery_IDX, :]  # (10200, 4)

        train, num_train_tracklets, num_train_pids, num_train_imgs = self._process_data(train_names, track_train,
                                                                                        home_dir='bbox_train',
                                                                                        relabel=True,
                                                                                        min_seq_len=min_seq_len)
        # train 是一个列表，长度为 8298，每个元素是一个元组，元祖里面第一为是元祖，里面是该taracklet的图片的路径，第二个元素是该tracklet的id，第三个元素是该tracklet的摄像头id
        # video = self._process_train_data(train_names, track_train, home_dir='bbox_train', relabel=True,
        #                                  min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = self._process_data(test_names, track_query,
                                                                                        home_dir='bbox_test',
                                                                                        relabel=False,
                                                                                        min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = self._process_data(test_names,
                                                                                                track_gallery,
                                                                                                home_dir='bbox_test',
                                                                                                relabel=False,
                                                                                                min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")


        #这里一改就把视频变成了图片
        # train = self.transform(train)
        # query = self.transform(query)
        # gallery = self.transform(gallery)

        # self.train_videos = video
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_cams = 6
        self.num_query_cams = 6
        self.num_gallery_cams = 6
        self.num_train_vids = num_train_tracklets
        self.num_query_vids = num_query_tracklets
        self.num_gallery_vids = num_gallery_tracklets
        self.datasets_name = 'mars'


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]  # 8298
        pid_list = list(set(meta_data[:, 2].tolist()))  # list 625
        num_pids = len(pid_list)
        if not relabel: pid2label = {pid: int(pid) for label, pid in enumerate(pid_list)}
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}  # 把离散的pid转换成连续的label 从0开始
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data  # 也就是mat中的四个值，分别是开始索引，结束索引，pid，camid
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            # if relabel: pid = pid2label[pid]
            pid = pid2label[pid]  # 利用上面的字典的映射关系，把pid转换成label
            camid -= 1  # index starts from 0
            img_names = names[
                        start_index - 1:end_index]  # 是一个list， 利用俩index 找到这个tracklet中的所有图片的名字，比如0001C1T0001F001.jpg，

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(
                set(camnames)) == 1, "Error: images are captured under different cameras!"  # 确认一个tracklets里面都是同一个摄像头拍摄的

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in
                         img_names]  # 按照mars目录的结构，把图片的路径拼接起来 如data/mars/bbox_train/0001/0001C1T0001F001.jpg
            keypoints_paths = [
                osp.join(self.root,
                         home_dir + '_keypoints',
                         img_name[:4],
                         img_name.replace('.jpg', '.pose'))  # 找到对应的tracklets下的这些图片的keypoints
                for img_name in img_names
            ]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)  # 把list转换成tuple
                tracklets.append((img_paths,keypoints_paths, pid,
                                  camid))  # 把一堆图片的路径，pid，camid放到一个tuple里面 (img_paths, pid, camid) 是一个tuple 里面有三个元素，第一个是该tracklets下图片的路径，第二个是pid，第三个是camid
                num_imgs_per_tracklet.append(len(img_paths))
                # if camid in video[pid] :
                #     video[pid][camid].append(img_paths)
                # else:
                #     video[pid][camid] =  img_paths

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet  # tracklets 是一个8298的list，里面的元素是tuple，tuple里面有三个元素，第一个是该tracklets下图片的路径，第二个是pid，第三个是camid

    def _process_train_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        video = defaultdict(dict)

        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"
            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                if camid in video[pid]:
                    video[pid][camid].extend(img_paths)
                else:
                    video[pid][camid] = img_paths
        return video
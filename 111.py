import torch
import numpy as np
from sklearn.linear_model import RANSACRegressor

def ransac_mean(tensor_list, percentage):
    data = np.stack([t.numpy() for t in tensor_list])  # 将tensor列表转换为numpy数组
    n_samples, n_features = data.shape

    if percentage == 1.0:
        # 如果百分比为100%，计算所有张量的平均值
        mean_tensor = torch.mean(torch.stack(tensor_list), dim=0)
    else:
        # 如果百分比小于100%，使用RANSAC算法选择内点
        # 调整RANSAC的阈值参数，使其选择指定百分比的数据点
        n_inliers = int(n_samples * percentage)
        mean_features = []
        for i in range(n_features):
            feature_values = data[:, i]
            X = np.arange(len(feature_values)).reshape(-1, 1)  # 创建X值（样本索引）
            y = feature_values  # 创建y值（特征值）
            ransac = RANSACRegressor(min_samples=n_inliers)
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            inliers = feature_values[inlier_mask]
            mean_feature = np.mean(inliers)
            mean_features.append(mean_feature)
        mean_tensor = torch.tensor(mean_features)

    return mean_tensor

# 假设您的tensor列表名为tensor_list
tensor_list = [torch.rand(1024) for _ in range(100)]  # 创建一个示例列表

# 调用函数，百分比为50%
representative_tensor = ransac_mean(tensor_list, 0.5)

import numpy as np
import matplotlib.pyplot as plt
from timm.data.random_erasing import RandomErasing
import torch

# 创建一个简单的图像
image = np.zeros((100, 100, 3))
image[:50, :, :] = [0, 0, 255]  # 蓝色的天空
image[50:, :, :] = [0, 255, 0]  # 绿色的草地

# 创建RandomErasing类的实例
random_erasing = RandomErasing(probability=0.5,mode='pixel', max_count=1, device='cpu')

# 显示原始图像和擦除后的图像
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.astype(np.uint8))
ax[0].set_title("Original Image")

# 将numpy数组转换为PyTorch张量
tensor_image = torch.tensor(image.transpose(2, 0, 1)).float()  # PyTorch通常使用浮点张量

# 使用RandomErasing对图像进行处理
erased_image = random_erasing(tensor_image)

# 将PyTorch张量转回为numpy数组
erased_image_np = erased_image.numpy().transpose(1, 2, 0)

ax[1].imshow(erased_image_np.astype(np.uint8))
ax[1].set_title("RandomErasing Applied")

plt.show()
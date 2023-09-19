def transform(input_list):
    result = []

    # 遍历列表中的每个元组
    for item in input_list:
        # 提取.jpg文件路径的元组和其他元素
        jpgs, *others = item
        # 对于每个.jpg文件路径，创建一个新的元组，添加额外的1，并添加到结果列表中
        for jpg in jpgs:
            result.append((jpg, *others, 1))

    return result


# 使用示例
input_list = [
    (('../data/MARS/bbox_train/0001/0001C1T0001F001.jpg', '../data/MARS/bbox_train/0001/0001C1T0001F002.jpg'), 0, 0),
    (('../data/MARS/bbox_train/0002/0002C1T0001F001.jpg', '../data/MARS/bbox_train/0002/0002C1T0001F002.jpg'), 1, 1)
]

output_list = transform(input_list)
print(output_list)

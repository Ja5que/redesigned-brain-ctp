# import os
#
# data_path = r"/mnt/e/dataset/Brain/net_data/train"
# data = os.listdir(data_path)
import os

# a = torch.rand(1, 2, 3)
# downsample_ct = F.interpolate(a, size=(16, 512, 512), mode='bilinear', align_corners=True)

# 假设输入是大小为 (30, 512, 512) 的张量
# input_tensor = torch.rand(1, 1, 30, 512, 512)
#
# # 调整输入张量的大小为 (16, 512, 512)
# output_tensor = F.interpolate(input_tensor, size=(16, 512, 512), mode='trilinear', align_corners=False)
# import numpy as np
# import torch
# data_upsample = torch.tensor([130, 512, 512])
# data = np.zeros([15, 512, 512])
#
# data_upsample[::2] = data[::1]
#
# print(np.sum(data_upsample[3] == data[2]))
import pydicom
import re
def natural_sort_key(s):
    """
    按文件名的结构排序，即依次比较文件名的非数字和数字部分
    """
    # 将字符串按照数字和非数字部分分割，返回分割后的子串列表
    sub_strings = re.split(r'(\d+)', s)
    # 如果当前子串由数字组成，则将它转换为整数；否则返回原始子串
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    # 根据分割后的子串列表以及上述函数的返回值，创建一个新的列表
    # 按照数字部分从小到大排序，然后按照非数字部分的字典序排序
    return sub_strings
# 路径到你的 DICOM 文件
dicoms_path = '/mnt/e/dataset/Brain/Rawdata/second_R/1/P8'
dicoms = sorted(os.listdir(dicoms_path), key=natural_sort_key)
for dicom_name in dicoms:
    ds = pydicom.dcmread(os.path.join(dicoms_path, dicom_name))
    print("Instance Number:", ds.get('InstanceNumber', 'No Instance Number found'))
# 读取 DICOM 文件

# 打印 Instance Number



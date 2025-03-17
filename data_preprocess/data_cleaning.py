# this file deal the data, including extract the clean data(good register data) and remove the bone
import shutil
import numpy as np
import pandas as pd
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from crip.io import *
import re
# extract the clean data
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
encoding_list = ['utf-8', 'gbk']

import os
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import pydicom
from PIL import Image
from crip.io import *
# 读取DICOM序列
class Dicom:
    def __init__(self, save_path, patient_info):
        self.save_path = save_path
        self.patient_info = patient_info

    def dicom2nii(self, dicom_dir, time=None):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, os.path.join(self.save_path, '%s_%s.nii.gz') % (self.patient_info, time))

    def dicom_to_tiff(self, dicom_path, time):
        # 读取 DICOM 文件
        files = []
        dicom_files = sorted(os.listdir(dicom_path), key=natural_sort_key)
        for filename in dicom_files:
            if filename.endswith(".dcm"):
                # print(filename)
                # 完整的文件路径
                filepath = os.path.join(dicom_path, filename)
                # 读取DICOM文件
                ds = pydicom.dcmread(filepath)
                # 将DICOM文件添加到列表中
                files.append(ds)
                # 获取图像数组
        files.sort(key=lambda x: float(x.get('InstanceNumber', 'No Instance Number found')))
        # image_array = ds.pixel_array
        # print(files)
        image_stack = np.stack([file.pixel_array for file in files])

        imwriteTiff(image_stack, os.path.join(self.save_path, '%s_%s.tiff') % (self.patient_info, time))

def extract_clean_data(data_path: str,
                       txt_path: str) -> list:
    df = pd.read_excel(txt_path)
    row_as_list = df.values.tolist()  # ['zyj', 1, nan, nan, nan, nan, nan, nan]
    data_path_list = []
    for i in range(len(row_as_list)):
        name = str(row_as_list[i][0])
        if_registration = row_as_list[i][1]
        patient_path = os.path.join(data_path, name)
        dicoms_files = os.listdir(patient_path)
        if if_registration == 0:
            try:
                regis_dicoms_files = [x for x in dicoms_files if x not in str(row_as_list[i][2:])]
            except:
                print(f"{name} all registration file is good")
        else:
            regis_dicoms_files = dicoms_files
        regis_dicoms_files_path = [os.path.join(patient_path, x) for x in regis_dicoms_files]
        data_path_list.append(regis_dicoms_files_path)
        print(f"processing {row_as_list[i][0]}, delete_file_num: {len(dicoms_files) - len(regis_dicoms_files)}")
    return data_path_list


if __name__ == '__main__':
    data_path_list = extract_clean_data(data_path=r'/mnt/e/dataset/Brain/Rawdata/first_R',
                                        txt_path=r'/mnt/e/dataset/Brain/Rawdata/frist_if_r.xlsx')
    for i in range(len(data_path_list)):
        patient_name = data_path_list[i][0].split('/')[-2]
        save_path = os.path.join(r'/mnt/e/dataset/Brain/3D_cleandata', patient_name)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        maybe_mkdir_p(save_path)
        processor = Dicom(save_path=save_path, patient_info=patient_name)
        for j in range(len(data_path_list[i])):
            time = data_path_list[i][j].split('/')[-1]
            print(f'processing {patient_name} and time {time}')
            processor.dicom_to_tiff(data_path_list[i][j], time)

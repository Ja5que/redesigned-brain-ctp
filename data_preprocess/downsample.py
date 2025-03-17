import os
import torch
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from crip.io import *
import shutil
def downsample_takehalf(img_path, save_path):
    # itk_ct = sitk.ReadImage(img_path)
    itk_name = img_path.split('/')[-1]
    # npy_ct = sitk.GetArrayFromImage(itk_ct)
    npy_ct = imreadTiff(img_path)
    d_npy_ct = npy_ct[::2]
    print(d_npy_ct.shape)
    imwriteTiff(d_npy_ct, os.path.join(save_path, itk_name))
    # d_itk_ct = sitk.GetImageFromArray(d_npy_ct)
    # d_itk_ct.SetOrigin(itk_ct.GetOrigin())
    # d_itk_ct.SetDirection(itk_ct.GetDirection())
    # d_itk_ct.SetSpacing(itk_ct.GetSpacing())
    # print(os.path.join(save_path, itk_name))
    # sitk.WriteImage(d_itk_ct, os.path.join(save_path, itk_name))
def get_mask(img_path, save_path):
    # itk_ct = sitk.ReadImage(img_path)
    itk_name = img_path.split('/')[-1]
    # print(itk_ct.GetSize())
    # sitk.WriteImage(itk_ct, os.path.join(save_path, itk_name))
    shutil.copy(img_path, os.path.join(save_path, itk_name))

if __name__ == '__main__':
    cts_path = r"/mnt/e/dataset/Brain/3D_cleandata"
    save_data_path = r"/mnt/e/dataset/Brain/net_data/data_takehalf_Tif"
    save_mask_path = r'/mnt/e/dataset/Brain/net_data/mask_Tif'
    maybe_mkdir_p(save_data_path)
    maybe_mkdir_p(save_mask_path)
    patients = os.listdir(cts_path)
    for patient in patients:
        patient_path = os.path.join(cts_path, patient)
        itk_cts = os.listdir(patient_path)
        for itk_ct in itk_cts:
            print(f'{patient} {itk_ct} processing')
            itk_ct_path = os.path.join(patient_path, itk_ct)
            downsample_takehalf(itk_ct_path, save_data_path)
            get_mask(itk_ct_path, save_mask_path)





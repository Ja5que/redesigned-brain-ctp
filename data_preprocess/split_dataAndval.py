import os
from sklearn.model_selection import train_test_split
import shutil
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

base_path = r'/root/autodl-tmp/net_data'

all_file_path = os.path.join(base_path, 'data_takehalf_Tif')
all_mask_path = os.path.join(base_path, 'mask_Tif')

train_file_path = os.path.join(base_path, 'train')
val_file_path = os.path.join(base_path, 'val')
train_mask_path = os.path.join(base_path, 'train_mask')
val_mask_path = os.path.join(base_path, 'val_mask')


# all_file_path = r'/mnt/e/dataset/Brain/net_data/data_takehalf_Tif'
# all_mask_path = r"/mnt/e/dataset/Brain/net_data/mask_Tif"

# train_file_path = r"/mnt/e/dataset/Brain/net_data_Tif/train"
# val_file_path = r"/mnt/e/dataset/Brain/net_data_Tif/val"
# train_mask_path = r"/mnt/e/dataset/Brain/net_data_Tif/train_mask"
# val_mask_path = r"/mnt/e/dataset/Brain/net_data_Tif/val_mask"


maybe_mkdir_p(train_file_path)
maybe_mkdir_p(val_mask_path)
maybe_mkdir_p(val_file_path)
maybe_mkdir_p(train_mask_path)


data = os.listdir(all_file_path)
u_set = set()
for ct in data:
    patient = ct.split('_')[0]
    u_set.add(patient)

patient_list = list(u_set)

train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    train_test_split(patient_list, patient_list, test_size=0.3, random_state=39)

print(f'Number of train {len(train_mask_paths)}, Number of val {len(val_mask_paths)}')
print("generate train set")
for train_p in train_img_paths:
    print("train processing {}".format(train_p))
    patient_all_ct = [x for x in data if x.split('_')[0] == train_p]
    patient_all_path = [os.path.join(all_file_path, x) for x in patient_all_ct]
    patient_all_m_path = [os.path.join(all_mask_path, x) for x in patient_all_ct]
    for patient_path, patient_m_path in zip(patient_all_path, patient_all_m_path):
        shutil.copy(patient_path, os.path.join(train_file_path, patient_path.split('/')[-1]))
        shutil.copy(patient_m_path, os.path.join(train_mask_path, patient_m_path.split('/')[-1]))

print("generate val set")
for val_p in val_img_paths:
    print("val processing {}".format(val_p))
    patient_all_ct = [x for x in data if x.split('_')[0] == val_p]
    patient_all_path = [os.path.join(all_file_path, x) for x in patient_all_ct]
    patient_all_m_path = [os.path.join(all_mask_path, x) for x in patient_all_ct]
    for patient_path, patient_m_path in zip(patient_all_path, patient_all_m_path):
        shutil.copy(patient_path, os.path.join(val_file_path, patient_path.split('/')[-1]))
        shutil.copy(patient_m_path, os.path.join(val_mask_path, patient_m_path.split('/')[-1]))


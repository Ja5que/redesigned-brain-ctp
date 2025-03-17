import torch.nn
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import os
from tqdm import tqdm
import re
import numpy as np
import torch.nn.functional as F
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
class Brain_To(Dataset):
    def __init__(self, options, mode='train'):
        print('Get 15passes/30passes for image-to-image mapping. all needed.')
        self.mode = mode
        assert mode in ['train', 'val'], f'{mode} is not in [train/val]'
        if self.mode == 'train':
            self.data_dir = options.train_data_path
            self.label_dir = options.train_label_path
        elif self.mode == 'val':
            self.data_dir = options.val_data_path
            self.label_dir = options.val_label_path

        self.patch_size = options.patch_size

        self.dataList = []
        self.labelList = []

        print(f'start loading {self.mode} data')
        cts = sorted(os.listdir(self.data_dir), key=natural_sort_key)
        labels = sorted(os.listdir(self.label_dir), key=natural_sort_key)

        for (ct, label) in tqdm(zip(cts, labels)):
            assert ct == label, f'{ct} and {label} do not match'
            img = sitk.ReadImage(os.path.join(self.data_dir, ct))
            npImg = sitk.GetArrayFromImage(img).astype('float')  # D H W the same is C H W

            lab = sitk.ReadImage(os.path.join(self.label_dir, label))
            npLab = sitk.GetArrayFromImage(lab).astype('long')  # D H W the same is C H W

            self.dataList.append(npImg)
            self.labelList.append(npLab)

        print(f'{self.mode}load done, length of dataset:', len(self.dataList))

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        image = self.dataList[idx]
        label = self.labelList[idx]

        max_v = label.max()
        min_v = label.min()

        image = (image - min_v) / (max_v - min_v)
        label = (label - min_v) / (max_v - min_v)

        tensor_image = torch.from_numpy(image).float()
        tensor_label = torch.from_numpy(label).float()


        tensor_image, tensor_label = self.randcrop(tensor_image, tensor_label)


        # assert tensor_image.shape == tensor_label.shape
        return {'data': tensor_image.float(), 'label': tensor_label.float()}

    def randcrop(self, img, label):
        _, H, W = img.shape

        w = self.patch_size[0]
        h = self.patch_size[1]

        diff_H = H - h
        diff_W = W - w

        if self.mode == 'train':
            rand_x = np.random.randint(0, diff_H)
            rand_y = np.random.randint(0, diff_W)

            croped_img = img[:, rand_x:rand_x + w, rand_y:rand_y + h]
            croped_lab = label[:, rand_x:rand_x + w, rand_y:rand_y + h]

        elif self.mode == 'val':
            # croped_img = img
            # croped_lab = label
        #
            center_x = W // 2
            center_y = H // 2

            croped_img = img[:,  center_x - w // 2:center_x + w // 2, center_y - h // 2:center_y + h // 2]
            croped_lab = label[:, center_x - w // 2:center_x + w // 2, center_y - h // 2:center_y + h // 2]

        return croped_img, croped_lab


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from option import options
    train_loader = DataLoader(dataset=Brain_To(options, mode='train'), batch_size=options.batch_size,
                              num_workers=options.num_workers, shuffle=True)
    # val_loader = DataLoader(dataset=Brain_To(options, mode='val'), batch_size=1,
    #                         num_workers=options.num_workers, shuffle=False)
    for i, data in enumerate(train_loader):
        data = data['data']
        B, D, H, W = data.shape
        data = data.view(B, 1, D, H, W)
        print(data.shape) # B C H W
        data_upsample = F.upsample(data, size=(30, 256, 256), mode='trilinear', align_corners=False)
        print(data_upsample.shape)
        for i in range(data.shape[2]):
            data_upsample[:, :,2*i] = data[:, :, i]
        data_upsample = data_upsample.squeeze(dim=1)
        break


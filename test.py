import numpy as np
import SimpleITK as sitk
from option import options
from model.model_2d import UNet2d, UNet2dRegis
import torch
import os
from model.Loss import *
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from crip.io import *
from tqdm import tqdm
import yaml
from model.TTUnet import TTUNet
# model_path = r"/mnt/no1/liuguannan/Brain/result/UNetRegis/best_model.pth"
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
model_path = config['model_test_path']
model_path = os.path.normpath(model_path)
res_path = config['res_test_path']
maybe_mkdir_p(res_path)
# res_path = os.path.normpath(res_path)
# res_path = os.path.join(options.res_path, model_path.split('/')[-2])
test_imgs = os.listdir(options.val_data_path)
test_label_path = os.listdir(options.val_label_path)
device = options.gpu

model = UNet2dRegis(in_chl=30, out_chl=30, model_chl=60).to(device)
# model = TTUNet(chl=15).to(device)
ssim = SSIM()
checkpoint = torch.load(model_path)  # , map_location='cpu'
model.load_state_dict(checkpoint['net'])

log = pd.DataFrame(index=[], columns=['name', 'ssim'])
input_path = os.path.join(res_path, 'input')
predict_path = os.path.join(res_path, 'test')
mark_path = os.path.join(res_path, 'mark')
maybe_mkdir_p(mark_path)
maybe_mkdir_p(predict_path)
maybe_mkdir_p(input_path)


model.eval()
for (t_name, m_name) in tqdm(zip(test_imgs, test_label_path)):
    assert t_name == m_name, '{} vs {}'.format(t_name, m_name)
    t_path = os.path.join(options.val_data_path, t_name)
    t_npy = imreadTiff(t_path)

    m_path = os.path.join(options.val_label_path, m_name)
    # m_ct = sitk.ReadImage(m_path)
    m_npy = imreadTiff(m_path)

    max_v = m_npy.max()
    min_v = m_npy.min()

    t_npy = (t_npy - min_v) / (max_v - min_v)
    m_npy = (m_npy - min_v) / (max_v - min_v)

    t_npy = t_npy[np.newaxis, :, :, :]
    m_npy = m_npy[np.newaxis, :, :, :]

    t_tensor = torch.FloatTensor(t_npy).to(device)  # 1, 15, 512, 512
    m_tensor = torch.FloatTensor(m_npy).to(device)
    with torch.no_grad():
        B, D, H, W = t_tensor.shape
        t_tensor = t_tensor.view(B, D, H, W)
        # t_upsample = F.upsample(t_tensor, size=(30, 512, 512), mode='trilinear', align_corners=False)
        # for i in range(t_tensor.shape[2]):
        #     t_upsample[:, :, 2 * i] = t_tensor[:, :, i]
        # t_upsample = t_upsample.squeeze(dim=1)
        t_upsample = t_tensor
        t_predict = model(t_upsample)
        t_predict = t_predict.view(B, D, H, W)  # restore original
        t_predict = model(t_upsample) # B D H W
        # print("t_predict shape: ", t_predict.shape)
        # print("t_tensor shape: ", t_tensor.shape)
        for i in range(15):
            t_predict[:, 2 * i] = t_tensor[:, i]
        score = ssim(t_predict, m_tensor)
        t_predict = np.squeeze(t_predict.data.cpu().numpy().clip(0, 1))
        t_input = np.squeeze(t_upsample.data.cpu().numpy().clip(0, 1))
    print("prdicting {}, ssim is {}".format(t_name, score.item()))

    t_predict = t_predict * (max_v - min_v) + min_v
    t_input = t_input * (max_v - min_v) + min_v
    print("t_predict shape: ", t_predict.shape)
    # ct_predcit = sitk.GetImageFromArray(t_predict)
    # ct_predcit.SetOrigin(m_ct.GetOrigin())
    # ct_predcit.SetSpacing(m_ct.GetSpacing())
    # ct_predcit.SetDirection(m_ct.GetDirection())
    mark = np.squeeze(m_tensor.data.cpu().numpy().clip(0, 1))
    mark = mark *(max_v - min_v) + min_v

    imwriteTiff(mark, os.path.join(mark_path, t_name.split('.')[0] + '_mark.tif'))
    # sitk.WriteImage(ct_predcit, os.path.join(predict_path, t_name.split('.')[0] + '_predict.nii.gz'))
    imwriteTiff(t_predict, os.path.join(predict_path, t_name.split('.')[0] + '_predict.tif'))
    imwriteTiff(t_input, os.path.join(input_path, t_name.split('.')[0] + '_input.tif'))
    tmp = pd.Series([t_name, score.item()], index=['name', 'ssim'])
    log = pd.concat([log, pd.DataFrame(tmp).T], axis=0, ignore_index=True)
log.to_csv(os.path.join(res_path, 'test.csv'))















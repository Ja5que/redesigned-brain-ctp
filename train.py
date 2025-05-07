import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from option import options
# from dataset.dataset import Brain_To
from dataset.dataset_server import Brain_To_Server_Tif
from model import weights_init
from model.model_2d import UNet2d, UNet2dRegis
import logger
import model.Loss as Loss
from model.Loss import SSIM
from datetime import datetime
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from utils import plot_loss_dice
import torch.nn.functional as F
import math
from model.TTUnet import TTUNet
train_loss_list = []
train_SSIM_list = []

val_loss_list = []
val_SSIM_list = []

# def param_para(batch_size):
#     rotate = nn.Parameter(torch.tensor(0.1 * math.pi / 180), requires_grad=True)
#     shift_x = nn.Parameter(torch.tensor(0.01), requires_grad=True)
#     shift_y = nn.Parameter(torch.tensor(0.01), requires_grad=True)
#
#     matrix = torch.tensor(
#         [[math.cos(math.radians(rotate)), -math.sin(math.radians(rotate)), shift_x],
#          [math.sin(math.radians(rotate)), math.cos(math.radians(rotate)), shift_y]],
#         dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
#     regis = nn.Parameter(matrix, requires_grad=True)
#
#     return regis



def val(model, val_loader, loss_mse, loss_ssim):
    val_loss = Loss.LossAverage()
    val_SSIM = Loss.LossAverage()

    model.eval()
    with torch.no_grad():
        for idx, pair in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = pair['data'].float(), pair['label'].float()
            data, target = data.to(device), target.to(device)

            B, D, H, W = data.shape
            data = data.view(B, 1, D, H, W)
            data_upsample = F.upsample(data, size=(30, 512, 512), mode='trilinear', align_corners=False)
            for i in range(data.shape[2]):
                data_upsample[:, :, 2 * i] = data[:, :, i]
            data_upsample = data_upsample.squeeze(dim=1)


            output = model(data_upsample)
            loss = loss_mse(output, target)  # 返回平均值
            ssim = loss_ssim(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_SSIM.update(ssim.item(), data.size(0))

    val_loss_list.append(val_loss.avg)
    val_SSIM_list.append(val_SSIM.avg)

    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_SSIM': val_SSIM.avg})
    info3 = "Val -- Loss: {:.3f}, SSIM: {:.3f} ".format(val_loss.avg, val_SSIM.avg)
    print(info3)
    return val_log


def train(model, train_loader, optimizer, loss_mse, loss_ssim, scheduler):
    train_loss = Loss.LossAverage()
    train_SSIM = Loss.LossAverage()

    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.param_groups[0]['lr']))

    model.train()
    for idx, pair in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = pair['data'].float(), pair['label'].float() # B C H W
        data, target = data.to(device), target.to(device)

        B, D, H, W = data.shape
        data = data.view(B, 1, D, H, W)
        data_upsample = F.upsample(data, size=(30, 512, 512), mode='trilinear', align_corners=False)
        for i in range(data.shape[2]):
            data_upsample[:, :,2*i] = data[:, :, i]
        data_upsample = data_upsample.squeeze(dim=1)

        optimizer.zero_grad()
        output = model(data_upsample)
        loss = loss_mse(output, target)  # 返回平均值
        ssim = loss_ssim(output, target)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), data.size(0))
        train_SSIM.update(ssim.item(), data.size(0))

    scheduler.step()

    train_loss_list.append(train_loss.avg)
    train_SSIM_list.append(train_SSIM.avg)

    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_SSIM': train_SSIM.avg})
    info2 = "Train -- Loss: {:.3f}, SSIM: {:.3f} ".format(train_loss.avg, train_SSIM.avg)
    print(info2)
    return train_log


if __name__ == '__main__':
    res_path = options.res_path
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    modelName = 'UNetRegis'
    save_path = os.path.join(res_path, modelName)
    maybe_mkdir_p(save_path)
    device = torch.device(options.gpu)
    
    print(res_path)
    print(save_path)
    print(device)
    # data info
    train_loader = DataLoader(dataset=Brain_To_Server_Tif(options, mode='train'), batch_size=options.batch_size,
                              num_workers=options.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=Brain_To_Server_Tif(options, mode='val'), batch_size=1,
                            num_workers=options.num_workers, shuffle=False, drop_last=True)

    # model info
    # model = UNet2dRegis(in_chl=30, out_chl=30, model_chl=60).to(device)
    model = TTUNet(chl=15).to(device)
    model.apply(weights_init.init_model)
    loss_mse = torch.nn.MSELoss()
    loss_ssim = SSIM()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    for epoch in range(1, options.epochs + 1):
        train_log = train(model, train_loader, optimizer, loss_mse, loss_ssim, scheduler)
        val_log = val(model, val_loader, loss_mse, loss_ssim)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        # torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_SSIM'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_SSIM']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
        if trigger >= 100:
            print("early stopping")
            break
        # # 深监督系数衰减
        # if epoch % 30 == 0: alpha *= 0.8
    plot_loss_dice(os.path.join(save_path, "train_log.csv"), save_path)

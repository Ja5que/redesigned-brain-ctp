import pandas as pd
import matplotlib.pyplot as plt
import os
def plot_loss_dice(csv_file_path, save_path):

    # 使用pandas读取CSV文件
    data = pd.read_csv(csv_file_path)

    # 将每一列数据存储为列表
    plt.figure(1)  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.plot(data['Train_Loss'], 'b',
             label='train_loss')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
    plt.plot(data['Val_Loss'], 'r', label='val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('train/val loss')
    plt.legend()  # 个性化图例（颜色、形状等）
    plt.savefig(os.path.join(save_path, "Loss.jpg"))

    plt.figure(2)  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.plot(data['Train_SSIM'], 'b',
             label='train_SSIM')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
    plt.plot(data['Val_SSIM'], 'r', label='val_SSIM')
    plt.ylabel('SSIM')
    plt.xlabel('epoch')
    plt.title('train/val SSIM')
    plt.legend()  # 个性化图例（颜色、形状等）
    plt.savefig(os.path.join(save_path, "SSIM.jpg"))




from crip.io import imreadTiff, imwriteTiff
import numpy as np
import os
import yaml
# from scipy.ndimage import zoom
import torch.nn.functional as F
import torch

# img is 15*512*512 tiff,need to interp to 30*256*256 tiff,the 15th slice should be the 29th slice
def interp_input_img(img_path, save_path):
    # read tiff
    img = imreadTiff(img_path)
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
    img_interp_256 = F.interpolate(img_tensor, size=(15, 256, 256), mode='trilinear', align_corners=False)
    img_interp = F.interpolate(img_interp_256, size=(30, 256, 256), mode='trilinear', align_corners=False)
    for i in range(img.shape[0]):
        img_interp[:,:,2 * i] = img_interp_256[:, :, i]
    img_interp = img_interp.squeeze(0).squeeze(0)  # remove batch and channel dimensions
    img_interp = img_interp.numpy()  # convert to numpy array
    # interp to 30*512*512
    # img_interp = np.zeros((30, 512, 512), dtype=np.float32)
    
    # for i in range(15):
    #     img_interp[2 * i] = img[i]
    # img_interp[29] = img[14]
    # #iterate slice 1,3 ... 27
    # for i in range(14):
    #     img_interp[2 * i + 1] = (img_interp[2 * i] + img_interp[2 * i + 2]) / 2
    # # interp 30*512*512 to 30*256*256 pic
    # img_interp =  zoom(img_interp, (1, 0.5, 0.5), order=1)  # linear interpolation
    
    imwriteTiff(img_interp, save_path)
# mask is 30*512*512 tiff, need to interp to 30*256*256 tiff   
def interp_mask_img(mask_path, save_path):
        # read tiff
    mask = imreadTiff(mask_path)
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
    mask_interp = F.interpolate(mask_tensor, size=(30, 256, 256), mode='trilinear', align_corners=False)
    mask_interp = mask_interp.squeeze(0).squeeze(0)  # remove batch and channel dimensions
    mask_interp = mask_interp.numpy()  # convert to numpy array
    # put mask into mask_interp
    # mask_interp = np.zeros((30, 512, 512), dtype=np.float32)
    # for i in range(30):
    #     mask_interp[i] = mask[i]
    # # interp 30*512*512 to 30*256*256 pic    

    # mask_interp = zoom(mask_interp, (1, 0.5, 0.5), order=0)  # nearest neighbor interpolation

    imwriteTiff(mask_interp, save_path)   
    
if __name__ == '__main__':
    # base_path = "D:/net_data_Tif/data_takehalf_Tif"
    # save_path = "D:/net_data_Tif/data_interp_Tif"
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    base_input_path = config['batch_interp_input_base_path']
    base_input_path = os.path.normpath(base_input_path)
    save_input_path = config['batch_interp_input_save_path']
    save_input_path = os.path.normpath(save_input_path)
    if not os.path.exists(save_input_path):
        os.makedirs(save_input_path)
        
    base_mask_path = config['batch_interp_mask_base_path']
    base_mask_path = os.path.normpath(base_mask_path)
    save_mask_path = config['batch_interp_mask_save_path']
    save_mask_path = os.path.normpath(save_mask_path)
    if not os.path.exists(save_mask_path):
        os.makedirs(save_mask_path)
    
    
    # emulate all the tiff files in base_path and interp them to save_path
    all_files = os.listdir(base_input_path)
    for file in all_files:
        # file_path = os.path.join(base_path, file)
        # save_file_path = os.path.join(save_path, file)
        # interp_img(file_path, save_file_path)
        # print(f'{file} interp done')
        file_path = os.path.join(base_input_path, file)
        save_file_path = os.path.join(save_input_path, file)
        interp_input_img(file_path, save_file_path)
        print(f'{file} interp done(input)')
        
    # emulate all the tiff files in base_mask_path and interp them to save_mask_path
    all_files = os.listdir(base_mask_path)
    for file in all_files:
        file_path = os.path.join(base_mask_path, file)
        save_file_path = os.path.join(save_mask_path, file)
        interp_mask_img(file_path, save_file_path)
        print(f'{file} interp done(mask)')
    
    

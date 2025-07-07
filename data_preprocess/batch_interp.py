from crip.io import imreadTiff, imwriteTiff
import numpy as np
import os
import yaml
# from scipy.ndimage import zoom
import torch.nn.functional as F
import torch


# get 30*512*512 tiff files
# interp them to 30*256*256 tiff files
# WARNING: this process should be done slice by slice, not all at once, the interpolation will broke original data
def make_mask_img_tif(input_path, save_path):
    # read tiff
    mask = imreadTiff(input_path)
    mask = mask.astype(np.float32)
    # interp 30*512*512 to 30*256*256, slice by slice
    mask_interp = np.zeros((30, 256, 256), dtype=np.float32)
    for i in range(30):
        mask_interp[i] = F.interpolate(torch.tensor(mask[i].reshape(1, 1, 512, 512)), 
                                        size=(256, 256), mode='bilinear', align_corners=False).squeeze().numpy()
    # save the interp result
    imwriteTiff(mask_interp, save_path)

# get 30*256*256 tiff files
# take half of the slices, 30*256*256 to 15*256*256
def make_takehalf_tif(input_path, save_path):
    #read tiff
    img = imreadTiff(input_path)
    img = img.astype(np.float32)
    # take half of the slices, 30*256*256 to 15*256*256
    img_takehalf = np.zeros((15, 256, 256), dtype=np.float32)
    for i in range(15):
        img_takehalf[i] = img[2 * i]
    # save the takehalf result
    imwriteTiff(img_takehalf, save_path)

# get 15*256*256 tiff files
# interp them to 30*256*256 tiff files
# remember to put the takehalf slices back to the interp result
def make_input_img_tif(input_path, save_path):
    # read tiff
    img = imreadTiff(input_path)
    img = img.astype(np.float32)
    # interp 15*256*256 to 30*256*256
    img_interp = np.zeros((30, 256, 256), dtype=np.float32)
    img_interp = F.interpolate(torch.tensor(img.reshape(1, 1, 15, 256, 256)),
                                size=(30, 256, 256), mode='trilinear', align_corners=False).squeeze().numpy()
    # put the takehalf slices back to the interp result
    for i in range(15):
        img_interp[2 * i] = img[i]
    # save the interp result
    imwriteTiff(img_interp, save_path)
                    
if __name__ == '__main__':
    
    
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    
    base_mask_path = config['batch_interp_mask_base_path']
    base_mask_path = os.path.normpath(base_mask_path)
    save_mask_path = config['batch_interp_mask_save_path']
    save_mask_path = os.path.normpath(save_mask_path)
    if not os.path.exists(save_mask_path):
        os.makedirs(save_mask_path)
    
    base_data_path = config['base_data_path']
    base_data_path = os.path.normpath(base_data_path)
    full_pass_data_path = base_data_path
    takehalfpass_data_path = config['takehalf_data_save_path']
    takehalfpass_data_path = os.path.normpath(takehalfpass_data_path)
    if not os.path.exists(takehalfpass_data_path):
        os.makedirs(takehalfpass_data_path)
    
    base_input_path = config['batch_interp_input_base_path']
    base_input_path = os.path.normpath(base_input_path)
    save_input_path = config['batch_interp_input_save_path']
    save_input_path = os.path.normpath(save_input_path)
    if not os.path.exists(save_input_path):
        os.makedirs(save_input_path)
    
    
    
    # # first, make mask files
    # # get all the tiff files in base_mask_path, in this case those were 30*512*512 tiff files
    # # interp them to 30*256*256 tiff files
    # # WARNING: this process should be done slice by slice, not all at once, the interpolation will broke original data
    # all_files = os.listdir(base_mask_path)
    # for file in all_files:
    #     file_path = os.path.join(base_mask_path, file)
    #     save_file_path = os.path.join(save_mask_path, file)
    #     make_mask_img_tif(file_path, save_file_path)
    #     print(f'{file} interp to mask done')
    
    
    # # second, make takehalf tiff files
    # # get all the tiff files in full_pass_data_path, in this case base_data_path equal to save_mask_path
    # # the tiff files are 30*256*256 tiff files, take half of the slices to 15*256*256 tiff files
    # all_files = os.listdir(full_pass_data_path)
    # for file in all_files: 
    #     file_path = os.path.join(full_pass_data_path, file)
    #     save_file_path = os.path.join(takehalfpass_data_path, file)
    #     make_takehalf_tif(file_path, save_file_path)
    #     print(f'{file} takehalf done')
    
    # third, interp the takehalf tiff files to 30*256*256 tiff files
    # get all the tiff files in base_input_path, in this case those were 15*256*256 tiff files
    # interp them to 30*256*256 tiff files
    # WARNING: the takehalf slices should be put back to the interp result
    all_files = os.listdir(base_input_path)
    for file in all_files:
        file_path = os.path.join(base_input_path, file)
        save_file_path = os.path.join(save_input_path, file)
        make_input_img_tif(file_path, save_file_path)
        print(f'{file} interp to input done')
    
    
    print("All done!")
    

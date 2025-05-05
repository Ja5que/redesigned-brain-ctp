from crip.io import imreadTiff, imwriteTiff
import numpy as np
import os
import yaml

# img is 15*512*512 tiff,need to interp to 30*512*512 tiff,the 15th slice should be the 29th slice
def interp_img(img_path, save_path):
    # read tiff
    img = imreadTiff(img_path)
    # interp to 30*512*512
    img_interp = np.zeros((30, 512, 512), dtype=np.float32)
    
    for i in range(15):
        img_interp[2 * i] = img[i]
    img_interp[29] = img[14]
    #iterate slice 1,3 ... 27
    for i in range(14):
        img_interp[2 * i + 1] = (img_interp[2 * i] + img_interp[2 * i + 2]) / 2
    
    imwriteTiff(img_interp, save_path)
    
    
    
if __name__ == '__main__':
    # base_path = "D:/net_data_Tif/data_takehalf_Tif"
    # save_path = "D:/net_data_Tif/data_interp_Tif"
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    base_path = config['batch_interp_base_path']
    base_path = os.path.normpath(base_path)
    save_path = config['batch_interp_save_path']
    save_path = os.path.normpath(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # emulate all the tiff files in base_path and interp them to save_path
    all_files = os.listdir(base_path)
    for file in all_files:
        file_path = os.path.join(base_path, file)
        save_file_path = os.path.join(save_path, file)
        interp_img(file_path, save_file_path)
        print(f'{file} interp done')
    
    
    

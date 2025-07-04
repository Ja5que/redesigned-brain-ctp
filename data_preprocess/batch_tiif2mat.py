import os
import yaml

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
base_path = config['base_path']
base_path = os.path.normpath(base_path)
res_path = config['to_mat_path']
res_path = os.path.normpath(res_path)
if not os.path.exists(res_path):
    os.makedirs(res_path)

data_path_tiff = os.path.join(base_path, 'mask_Tif')

# for each tiff in data_path_tiff, filename satisfies the pattern: [X]_P[Y].tiff
# X is the patient number, Y is the height of the slice relative to the patient
# each tiff has 30 slices, each slice is 512x512 picture
# we want to convert each tiff to matlab data format .mat
# each .mat file has the following structure:
# {
    # 'V': [30, 512, 512], # the 30 slices of the tiff
    # 'AIFy': a scalar, Y Location of Arterial Input Function, default is 160
    # 'AIFx': a scalar, X Location of Arterial Input Function, default is 266
    # 'VOFy': a scalar, Y Location of Venous Output Function, default is 466
    # 'VOFx': a scalar, X Location of Venous Output Function, default is 257
    # 'PRE': a scalar, Pre-enhancement cutoff slice, default is 1
    # 'POST': a scalar, Post-enhancement cutoff slice, default is 30
# }
# the .mat file is saved in res_path, remains the same filename

import numpy as np
import tifffile as tiff
import scipy.io as sio

for filename in os.listdir(data_path_tiff):
    if filename.endswith('.tiff'):
        print(filename)
        # remove the extension
        clean_filename = filename.split('.')[0]
        # read the tiff file
        tiff_data = tiff.imread(os.path.join(data_path_tiff, filename))
        # convert to numpy array, shape: [30, 512, 512] and type: double
        tiff_data = np.array(tiff_data, dtype=np.double)
        # save to .mat file
        data = {
            'V': tiff_data,
            'AIFy': 160,
            'AIFx': 266,
            'VOFy': 466,
            'VOFx': 257,
            'PRE': 5,
            'POST': 30
        }
        sio.savemat(os.path.join(res_path, clean_filename + '.mat'), data)
        print('saved to', os.path.join(res_path, clean_filename + '.mat'))

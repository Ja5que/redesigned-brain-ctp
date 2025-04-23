import os
import yaml
import numpy as np
import tifffile as tiff
import scipy.io as sio

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
base_path = config['base_path']
base_path = os.path.normpath(base_path)
res_path = config['res_path']
res_path = os.path.normpath(res_path)



mat_name = os.path.join(base_path,'ctp.mat')
mat = sio.loadmat(mat_name)
V = mat['V']

# convert V to tiff and save to res_path
res_path = os.path.join(res_path, 'ctp')
if not os.path.exists(res_path):
    os.makedirs(res_path)
for i in range(V.shape[0]):
    tiff.imsave(os.path.join(res_path, str(i) + '.tiff'), V[i])
print('saved to', res_path)


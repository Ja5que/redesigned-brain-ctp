import os
import yaml
# for file in config['base_path'] check tiff size whether same
import tifffile as tiff
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
base_path = config['base_path']
base_path = os.path.normpath(base_path)

#emulate folder in base_path
for folder in ['data_takehalf_Tif', 'mask_Tif', 'train', 'val', 'train_mask', 'val_mask']:
    # maybe_mkdir_p(os.path.join(base_path, folder))
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        continue
    print(f"Checking files in {folder_path}")
    files = os.listdir(folder_path)
    # get random tiff in folder,get its shape,check all tiff in folder have same shape
    if not files:
        print(f"No files found in {folder_path}.")
        continue
    first_file = files[0]
    first_file_path = os.path.join(folder_path, first_file)
    if not first_file.endswith('.tif') and not first_file.endswith('.tiff'):
        print(f"File {first_file_path} is not a tiff file.")
        continue
    first_shape = tiff.imread(first_file_path).shape
    print(f"First file shape: {first_shape}")
    for file in files:
        file_path = os.path.join(folder_path, file)
        if not file.endswith('.tif') and not file.endswith('.tiff'):
            print(f"File {file_path} is not a tiff file.")
            continue
        shape = tiff.imread(file_path).shape
        if shape != first_shape:
            print(f"File {file_path} has different shape: {shape}, expected: {first_shape}")
        # else:
        #     print(f"File {file_path} has the same shape: {shape}")
    print(f"All files in {folder_path} have been checked.")
    print(f"Finished checking folder: {folder_path}")
print("All folders have been checked.")
    
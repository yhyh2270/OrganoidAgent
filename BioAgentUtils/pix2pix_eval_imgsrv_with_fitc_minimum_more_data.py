# Auto-generated from pix2pix_eval_imgsrv_with_fitc_minimum_more_data.ipynb
# Source: /home/lachlan/ProjectsLFS/OrganoidAgent/pixel2pixel_fluorescent/fluorescent/pix2pix_eval_imgsrv_with_fitc_minimum_more_data.ipynb

# %% [cell 1]
import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io

from pix2pix_modules import GeneratorUNet as Generator, Discriminator

from IPython.display import display, clear_output
from torch.utils.data import random_split

import random


# os.environ["CUDA_VISIBLE_DEVICES"]="1"

num_devices = torch.cuda.device_count()
for i in range(num_devices):
    device = torch.device(f'cuda:{i}')
    name = torch.cuda.get_device_name(device)
    print(f'Device {i}: {name}')

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set the current device
torch.cuda.set_device(device)
device_current = torch.cuda.current_device()
print('Current device:', device_current)
print('Device name:', torch.cuda.get_device_name(device_current))

# %% [cell 2]
# !pip install pymysql

# %% [cell 3]
# example_input = torch.rand(64, 1, 256, 256).to(device)

# generator = Generator(in_channels=1, out_channels=1).to(device)
# generator.load_state_dict(torch.load(os.path.join(checkpoint_dir_pretrained, 'best_generator.pth')))

# traced_script_module = torch.jit.trace(generator, example_input)

# os.makedirs("jit-models", exist_ok=True)

# torch.jit.save(traced_script_module, "tritc-xsp-wo-pretrain.pt")

# %% [cell 4]
class NormalizeImage:
    def __call__(self, pic):
        img = pic.astype(np.float32)
        # img = img / 65535.
        img = np.transpose(img, (2, 0, 1)) if len(img.shape) == 3 else img[np.newaxis, :, :]
        return torch.from_numpy(img)

# %% [cell 5]
class GridFixedCrop:
    def __init__(self, grid_size, crop_size, color_bit, maximum):
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.normalize = NormalizeImage()
        self.divisor = (2**color_bit) - 1
        self.maximum = maximum

    def __call__(self, img):
        w, h = img.shape[:2]
        stride = w // self.grid_size
        crops = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                top = i * stride
                left = j * stride
                
                # Center the crop in each cell of the grid
                if stride > self.crop_size:
                    top += (stride - self.crop_size) // 2
                    left += (stride - self.crop_size) // 2
                
                crop = img[top: top + self.crop_size, left: left + self.crop_size]
                crops.append(self.normalize(crop))
        
        # return torch.stack(crops) / self.maximum
        
        crops = torch.stack(crops) / self.maximum
        if self.maximum == -1:
            crops = (crops - crops.min()) / max(1e-6, crops.max() - crops.min())
        
        return crops

# %% [cell 6]
# Instantiate your cropping tool
grid_size = 8  # replace with your grid size
crop_size = 256  # replace with your crop size
color_bit = 16  # replace with your color bit
# maximum = 2 ** color_bit - 1  # replace with your maximum value
# maximum = max_trans = 51657.95 
# max_fluo = 714.0499999999993
maximum = max_trans = -1
max_fluo = 1000
cropper = GridFixedCrop(grid_size, crop_size, color_bit, maximum)

# %% [cell 7]
import pymysql

sql_connect = {
    "host": os.environ.get("IMGSERV_DB_HOST", "localhost"),
    "user": os.environ.get("IMGSERV_DB_USER", "root"),
    "password": os.environ.get("IMGSERV_DB_PASSWORD"),
    "db": os.environ.get("IMGSERV_DB_NAME", "image_server"),
}

if not sql_connect["password"]:
    raise RuntimeError("Set IMGSERV_DB_PASSWORD before connecting to the image server")

conn = pymysql.connect(
    host=sql_connect['host'], 
    user=sql_connect['user'], 
    password=sql_connect['password'], 
    db=sql_connect['db']
)


def get_image_paths(conn, project_id, channel):
    
    data_root = "/media/DATA/image_server/"
    with conn.cursor() as cur:
        query = f"SELECT merged_path, well FROM imgsrv_images_merged WHERE project_id={project_id} AND channel='{channel}'"
        cur.execute(query)
        results = cur.fetchall()
        return [(os.path.join(data_root, result[0]), result[1]) for result in results]


from skimage import io

def read_image(image_path):
    # Add the necessary code to read the image from the path
    image = io.imread(image_path)
    # Add any necessary pre-processing steps
    return image

# %% [cell 8]
get_image_paths(conn, 2486, "TL")

# %% [cell 9]
def stitch_crops(crops, grid_size, crop_size):
    """
    Function to stitch together image crops into a single large image.

    Parameters:
    - crops: a list or a tensor of crops, should have length grid_size*grid_size.
    - grid_size: the number of crops along one dimension of the image grid.
    - crop_size: the size of one crop along one dimension (crops are square).
    """
    # Ensure that crops is a numpy array
    crops = crops.cpu().detach().numpy() if isinstance(crops, torch.Tensor) else np.array(crops)

    # Get the number of channels (should be 1 for grayscale, 3 for RGB)
    n_channels = crops.shape[1]

    # Initialize an empty array to hold the final stitched image
    full_img_size = grid_size * crop_size
    stitched_img = np.zeros((full_img_size, full_img_size, n_channels))

    # Reshape crops if necessary
    crops = crops.reshape((grid_size, grid_size, crop_size, crop_size, n_channels))

    # Iterate over each crop and place it in the correct position in the stitched image
    for i in range(grid_size):
        for j in range(grid_size):
            top = i * crop_size
            left = j * crop_size

            stitched_img[top:top+crop_size, left:left+crop_size] = crops[i, j]

    return stitched_img

# %% [cell 10]
def change_file_name(file_path, new_suffix):
    # Split the file path into directory and file name
    directory, filename = os.path.split(file_path)

    # Split the filename into base and extension
    base, extension = os.path.splitext(filename)

    # Replace the last part of the base (after "_") with the new suffix
    # new_base = base.rsplit("_", 1)[0] + "_" + new_suffix
    new_base = f"{base}_{new_suffix}"

    # Return the new file path
    return os.path.join(directory, new_base + extension)

# %% [cell 11]
# /media/DATA/image_server/thumbnail/MDSCREEN/XSP/GAS23317-P-T0-noCalcein-M_Plate_20537/TimePoint_1/GAS23317-P-T0-noCalcein-M_PseudoFITC_thumbnail.png

# %% [cell 12]
def get_thumbnail_path(row):
    # path_text=[i for i in row.index if "_path" in str(i)]
    src_path=row['merged_path']

    dst_path=src_path.replace("PSEUDO", "thumbnail")
    os.makedirs(os.path.dirname(dst_path),exist_ok=True)

    r, ext = os.path.splitext(dst_path)
    local_path = r+'_thumbnail.png'
    print(local_path)

    image=io.imread(src_path)
    ratio=1

    from datetime import datetime
    now=datetime.now().strftime('%Y-%m-%d/%H:%M:%S')
    os.makedirs(f"temp/{now}", exist_ok=True)
    temp_path = os.path.join("temp/", now,os.path.basename(src_path))
    # try:
    if len(image.shape)==3:
        if image.shape[-1]==3 or image.shape[-1]==4:
            if not os.path.exists(local_path):
                os.system(f"ffmpeg -hide_banner -loglevel error -y -i '{src_path}' -vf scale=-1:512 '{local_path}'")
            ratio=image.shape[-2]/512
        else:
            if not os.path.exists(temp_path):
                df = rescale(image)
                io.imsave( temp_path,df.astype(np.uint16) if np.max(df) >256 else df.astype(np.uint8),check_contrast=False)
            if not os.path.exists(local_path):
                os.system(f"ffmpeg -hide_banner -loglevel error -y -i '{temp_path}' -vf scale=-1:512 '{local_path}'")
            ratio = image.shape[-1] / 512
    elif len(image.shape) == 2:
        if not os.path.exists(temp_path):
            df=rescale(image)
            io.imsave(temp_path,df.astype(np.uint16) if np.max(df) >256 else df.astype(np.uint8),check_contrast=False)
        if not os.path.exists(local_path):
            os.system(f"ffmpeg -hide_banner -loglevel error -y -i '{temp_path}' -vf scale=-1:512 '{local_path}'")
        ratio = image.shape[-1] / 512
    return local_path, ratio


    # except:
    #     pass
    #     return  None,None

def rescale(img):
    img = img.copy().astype(float)  # Use float instead of np.float
    img = img - img.min()
    img = img / (img.max() if img.max() > 0 else 1)

    return (img * 255).astype(np.uint8)

# %% [cell 13]
def insert_or_update_image(conn, project_id, channel, merged_path, well, base_dir="/media/DATA/image_server/"):
    thumbnail_path, _ = get_thumbnail_path({'merged_path': merged_path})  # unpack ratio, which is not used
    with conn.cursor() as cur:
        cur.execute(f"""SELECT project_name, operation, plate_name, time_point, well, stage, site_x, site_y, field, image_type, color, is_merged, rescale_ratio, is_z_projected, z_project_method, is_stitched FROM imgsrv_images_merged WHERE project_id={project_id} AND channel='TL'""")
        result = cur.fetchone()
        if result is None:
            print(f"No data found for project_id {project_id}")
            return

        project_name, operation, plate_name, time_point, _, stage, site_x, site_y, field, image_type, color, is_merged, rescale_ratio, is_z_projected, z_project_method, is_stitched = (result + (None,) * 16)[:16]

        if field is None:
            field = -1  # or any other default integer value

        
        cur.execute(f"""INSERT INTO imgsrv_images_merged (project_name, project_id, channel, merged_path, thumbnail_path, operation, plate_name, time_point, well, stage, site_x, site_y, field, image_type, color, is_merged, rescale_ratio, is_z_projected, z_project_method, is_stitched)
                        VALUES ('{project_name}', {project_id}, '{channel}', '{os.path.relpath(merged_path, base_dir)}', '{os.path.relpath(thumbnail_path, base_dir)}', '{operation}', '{plate_name}', {time_point}, '{well}', '{stage}', {site_x}, {site_y}, '{field}', '{image_type}', '{color}', {is_merged}, {rescale_ratio}, {is_z_projected}, '{z_project_method}', {is_stitched})
                        ON DUPLICATE KEY UPDATE
                        merged_path = VALUES(merged_path),
                        thumbnail_path = VALUES(thumbnail_path),
                        operation = VALUES(operation),
                        plate_name = VALUES(plate_name),
                        time_point = VALUES(time_point),
                        well = VALUES(well),
                        stage = VALUES(stage),
                        site_x = VALUES(site_x),
                        site_y = VALUES(site_y),
                        field = VALUES(field),
                        image_type = VALUES(image_type),
                        color = VALUES(color),
                        is_merged = VALUES(is_merged),
                        rescale_ratio = VALUES(rescale_ratio),
                        is_z_projected = VALUES(is_z_projected),
                        z_project_method = VALUES(z_project_method),
                        is_stitched = VALUES(is_stitched)""")
    conn.commit()

# %% [cell 14]
def mark_as_processed(conn, project_id):
    with conn.cursor() as cur:
        query = f"INSERT IGNORE INTO imgsrv_processed_ids (project_id) VALUES ({project_id})"
        cur.execute(query)
        conn.commit()

# %% [cell 15]
import torch
from torchvision import transforms

# Load the TorchScript model.
# loaded_model = torch.jit.load("jit-models/tritc-xsp-wo-pretrain.pt")
loaded_model = torch.jit.load("jit-models/fitc-minimum-more-data.pt")

# %% [cell 16]

# %% [cell 17]
# Given a project_id and channel, get the list of image paths
# project_id = 2278  # replace with your project_id
# project_id = 2332
channel = 'TL'  # replace with your channel

# project_ids = [2332]
# project_ids = [2319,2329,2326,2320,2318,2492,2491,2490,2488]
project_ids = [2529]

for project_id in project_ids:
    image_paths = get_image_paths(conn, project_id, channel)

    # Iterate over the images
    for image_path, well in image_paths:
        # Read the image
        image = read_image(image_path)

        # Crop the image into a batch of crops
        image_crops = cropper(image).to(device)

        # # Convert the image to a tensor
        # # Assuming your model expects a 4D tensor (batch_size, channels, height, width)
        # # and the image is grayscale and needs to be normalized to [0, 1]
        # image_tensor = transforms.ToTensor()(image).unsqueeze(0)

        # Perform inference
        output_data = loaded_model(image_crops)

        fluo_stitched = stitch_crops(output_data, grid_size, crop_size)

        # Define the new image path. 
        # This will replace the first directory in the path after '/image_server/' with 'PSEUDO'
        parts = image_path.split('/')
        parts[parts.index('image_server') + 1] = 'PSEUDO'
        new_image_path = '/'.join(parts)

        new_image_path = change_file_name(new_image_path, 'PseudoFITC')
        fluo_stitched = np.squeeze(fluo_stitched)



        # Define the new directory and create it if it doesn't exist
        new_directory = os.path.dirname(new_image_path)
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        # Save the image
        io.imsave(new_image_path, (fluo_stitched*max_fluo).astype(np.uint16))


        insert_or_update_image(conn, project_id, "pseudoFITC", new_image_path, well)

mark_as_processed(conn, project_id)

# %% [cell 18]
os.path.relpath("/media/DATA/image_server/thumbnail/MDSCREEN/XSP/GAS23200-calcein-P0-Day13-t96_Plate_182/TimePoint_1/GAS23200-calcein-P0-Day13-t96_C10_PseudoFITC_thumbnail.png", "/media/DATA/image_server/")

# %% [cell 19]
# !ls thumbnail/media/DATA/image_server/PSEUDO/MDSCREEN/XSP

# %% [cell 20]
# !pwd

# %% [cell 21]
io.imread("/media/DATA/image_server/PSEUDO/MDSCREEN/XSP/GAS23200-calcein-P0-Day13-t96_Plate_182/TimePoint_1/GAS23200-calcein-P0-Day13-t96_B05_PseudoFITC.tif")

# %% [cell 22]
fluo_stitched.shape

# %% [cell 23]
fig = plt.figure(dpi=300)
io.imshow(image)

# %% [cell 24]
fluo_stitched.min()

# %% [cell 25]
fig = plt.figure(dpi=300)
io.imshow(fluo_stitched*max_fluo)

# %% [cell 26]

# %% [cell 27]
output_data.shape

# %% [cell 28]
image.shape

# %% [cell 29]
len(image_paths)

# %% [cell 30]

# %% [cell 31]

# %% [cell 32]

# %% [cell 33]

# %% [cell 34]

# %% [cell 35]

# %% [cell 36]

# %% [cell 37]

# %% [cell 38]

# %% [cell 39]

# %% [cell 40]

# %% [cell 41]

# %% [cell 42]

# %% [cell 43]

# %% [cell 44]
torch.manual_seed(0)  # Or any other seed

# %% [cell 45]
class NormalizeImage:
    def __call__(self, pic):
        img = pic.astype(np.float32)
        # img = img / 65535.
        img = np.transpose(img, (2, 0, 1)) if len(img.shape) == 3 else img[np.newaxis, :, :]
        return torch.from_numpy(img)

class GridFixedCrop:
    def __init__(self, grid_size, crop_size, color_bit, maximum):
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.normalize = NormalizeImage()
        self.divisor = (2**color_bit) - 1
        self.maximum = maximum

    def __call__(self, img):
        w, h = img.shape[:2]
        stride = w // self.grid_size
        crops = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                top = i * stride
                left = j * stride
                
                # Center the crop in each cell of the grid
                if stride > self.crop_size:
                    top += (stride - self.crop_size) // 2
                    left += (stride - self.crop_size) // 2
                
                crop = img[top: top + self.crop_size, left: left + self.crop_size]
                crops.append(self.normalize(crop))
        
        return torch.stack(crops) / self.maximum

    
class GridRandomCrop:
    def __init__(self, grid_size, crop_size, color_bit, maximum):
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.normalize = NormalizeImage()
        self.divisor = (2**color_bit) - 1
        self.maximum = maximum

    def __call__(self, img):
        w, h = img.shape[:2]
        stride = w // self.grid_size  
        crops = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                top = i * stride
                left = j * stride
                if stride > self.crop_size:
                    top += np.random.randint(0, stride - self.crop_size)
                    left += np.random.randint(0, stride - self.crop_size)
                crop = img[top: top + self.crop_size, left: left + self.crop_size]
                crops.append(self.normalize(crop)) 
        # return torch.stack(crops) / self.divisor
        return torch.stack(crops) / self.maximum


class FITCDataset(Dataset):
    def __init__(self, 
                 root_dir, save_dir, 
                 regenerate_transform=False, load_from_hdf5=True, read_from_disk=False, 
                 grid_size=8, trans_maximum=65535, fluoro_maximum=4095, subset_size=None, randomness=True):
        if randomness:
            GridCrop = GridRandomCrop
        else:
            GridCrop = GridFixedCrop
        
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.regen_transform = regenerate_transform
        self.load_from_hdf5 = load_from_hdf5
        self.read_from_disk = read_from_disk
        self.grid_size = grid_size

        self.file_list = glob.glob(os.path.join(root_dir, '*/*/*_w1.tif')) 
        
        if subset_size is not None:
            random.shuffle(self.file_list)
            self.file_list = self.file_list[:int(subset_size*len(self.file_list))]
        
        self.transform_transparent = GridCrop(grid_size, 256, 16, trans_maximum)
        self.transform_fluorescent = GridCrop(grid_size, 256, 12, fluoro_maximum)
        self.transformed_dir = os.path.join(save_dir, 'transformed')
        self.hdf5_file_path = os.path.join(save_dir, 'transformed_data.hdf5')
        self.crop_list = [(img_idx, crop_idx) for img_idx in range(len(self.file_list)) for crop_idx in range(grid_size*grid_size)]

        if self.regen_transform or not self.check_transform_exists():
            if not os.path.exists(self.transformed_dir):
                os.makedirs(self.transformed_dir)
            self.transform_and_save()

        if not self.load_from_hdf5:
            self.save_to_hdf5()

        if self.load_from_hdf5:
            if not self.check_hdf5_exists():
                print("HDF5 file does not exist. Regenerating...")
                self.save_to_hdf5()
            self.data = h5py.File(self.hdf5_file_path, 'r')
        elif self.read_from_disk:
            # Load transformed data from npy files on-the-fly
            self.data = [(os.path.join(self.transformed_dir, f'{i}_trans.npy'),
                          os.path.join(self.transformed_dir, f'{i}_fluo.npy'))
                         for i in range(len(self.file_list))]
        else:
            # Load all data into memory (This could consume a lot of memory)
            self.data = [(np.load(os.path.join(self.transformed_dir, f'{i}_trans.npy')),
                          np.load(os.path.join(self.transformed_dir, f'{i}_fluo.npy')))
                         for i in range(len(self.file_list))]
            
            
    def check_hdf5_exists(self):
        return os.path.isfile(self.hdf5_file_path)

    def check_transform_exists(self):
        # Check if there's at least one pair of transformed data (transparent and fluorescent images)
        transparent_exists = os.path.isfile(os.path.join(self.transformed_dir, '0_trans.npy'))
        fluorescent_exists = os.path.isfile(os.path.join(self.transformed_dir, '0_fluo.npy'))

        return transparent_exists and fluorescent_exists


    def transform_and_save(self):
        for img_idx in range(len(self.file_list)):
            transparent_image_path = self.file_list[img_idx]
            fluorescent_image_path = transparent_image_path.replace('_w1.tif', '_w2.tif')

            transparent_image = io.imread(transparent_image_path)

            # If the fluorescent image doesn't exist, just use the transparent one.
            if os.path.exists(fluorescent_image_path):
                fluorescent_image = io.imread(fluorescent_image_path)
            else:
                fluorescent_image = transparent_image

            transformed_transparent_image = self.transform_transparent(transparent_image)
            transformed_fluorescent_image = self.transform_fluorescent(fluorescent_image)

            # Save each crop separately
            for crop_idx in range(transformed_transparent_image.shape[0]):
                np.save(os.path.join(self.transformed_dir, f'{img_idx}_{crop_idx}_trans.npy'), transformed_transparent_image[crop_idx])
                np.save(os.path.join(self.transformed_dir, f'{img_idx}_{crop_idx}_fluo.npy'), transformed_fluorescent_image[crop_idx])



    def save_to_hdf5(self):
        with h5py.File(self.hdf5_file_path, 'w') as hf:
            for img_idx in range(len(self.file_list)):
                for crop_idx in range(self.grid_size * self.grid_size):
                    transformed_transparent_image = np.load(os.path.join(self.transformed_dir, f'{img_idx}_{crop_idx}_trans.npy'))
                    transformed_fluorescent_image = np.load(os.path.join(self.transformed_dir, f'{img_idx}_{crop_idx}_fluo.npy'))

                    hf.create_dataset(f'transparent_{img_idx}_{crop_idx}', data=transformed_transparent_image)
                    hf.create_dataset(f'fluorescent_{img_idx}_{crop_idx}', data=transformed_fluorescent_image)


    def __len__(self):
        return len(self.crop_list)

    def __getitem__(self, idx):
        img_idx, crop_idx = self.crop_list[idx]

        if self.load_from_hdf5:
            if self.read_from_disk:
                with h5py.File(self.hdf5_file_path, 'r') as hf:
                    transparent_image = hf[f'transparent_{img_idx}_{crop_idx}'][:]
                    fluorescent_image = hf[f'fluorescent_{img_idx}_{crop_idx}'][:]
            else:
                transparent_image = self.data[f'transparent_{img_idx}_{crop_idx}'][:]
                fluorescent_image = self.data[f'fluorescent_{img_idx}_{crop_idx}'][:]
        else:
            transparent_image = np.load(os.path.join(self.transformed_dir, f'{img_idx}_{crop_idx}_trans.npy'))
            fluorescent_image = np.load(os.path.join(self.transformed_dir, f'{img_idx}_{crop_idx}_fluo.npy'))

        return torch.from_numpy(transparent_image), torch.from_numpy(fluorescent_image)

# %% [cell 46]
max_trans = 51657.95 / 2
max_fluo = 714.0499999999993

# %% [cell 47]
# Hyperparameters
lr = 0.0002
batch_size = 64
epochs = 1000

# Define the datasets and dataloaders
root_dir = 'data-tritc-xsp'
save_dir = 'dataset-tritc-xsp'
dataset = FITCDataset(root_dir, save_dir, 
                      regenerate_transform=False, load_from_hdf5=True, read_from_disk=False,
                      trans_maximum=max_trans, fluoro_maximum=max_fluo, subset_size=None
                     )
# Define the proportion for the split
train_proportion = 0.8
test_proportion = 1 - train_proportion

# Calculate the number of samples for each set
num_train = int(train_proportion * len(dataset) / batch_size) * batch_size
num_test = len(dataset) - num_train

print("num_train: ", num_train)
print("num_test: ", num_test)

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

# %% [cell 48]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

# define data loaders for validation and test datasets
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=32)

# %% [cell 49]
# Define the datasets and dataloaders
root_dir_val = 'data-tritc-val'
save_dir_val = 'dataset-tritc-val'
dataset_val = FITCDataset(root_dir_val, save_dir_val, 
                      regenerate_transform=True, load_from_hdf5=False, read_from_disk=True,
                      trans_maximum=max_trans, fluoro_maximum=max_fluo, subset_size=10, randomness=False
                     )

# %% [cell 50]
num_train = int(len(dataset_val) * 0.8)
num_test = len(dataset_val) - num_train
print("num_train: ", num_train)
print("num_test: ", num_test)

# # Split the dataset
# val_dataset, _ = random_split(dataset_val, [num_train, num_test])
val_dataset = dataset_val

# %% [cell 51]
# define data loaders for validation and test datasets
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32)

# %% [cell 52]
checkpoint_dir = "checkpoints/pixel2pixel-tritc-xsp-wo-pt/"

# %% [cell 53]
import torch
from torch.autograd import Variable

# Define the generator and discriminator models, and optimizers
generator = Generator(in_channels=1, out_channels=1).to(device)
# discriminator = Discriminator(in_channels=1).to(device)
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load state dicts into the models and optimizers
generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_generator.pth')))
# discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_discriminator.pth')))
# optimizer_G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_optimizer_G.pth')))
# optimizer_D.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_optimizer_D.pth')))

# Remember to call model.eval() to set dropout and batch norm layers to evaluation mode
generator.eval()

# Now we can use generator to make predictions
def make_predictions(data_loader):
    all_predictions = []

    # Iterate through your data loader
    for batch in data_loader:
        inputs = batch[0]  # assuming your input data is the first element of your batch

        # If you're using a GPU, you need to send your input data to the GPU
        inputs = Variable(inputs).to(device)

        # Forward pass
        with torch.no_grad():  # This is important to deactivate autograd during evaluation
            outputs = generator(inputs)
            all_predictions.append(outputs.cpu())  # Assuming you want to gather all predictions and return
            
        break

    return all_predictions

predictions = make_predictions(val_loader)  # assuming val_loader is your DataLoader for the validation dataset

# %% [cell 54]
predictions[0].shape

# %% [cell 55]
def stitch_crops(crops, grid_size, crop_size):
    """
    Function to stitch together image crops into a single large image.

    Parameters:
    - crops: a list or a tensor of crops, should have length grid_size*grid_size.
    - grid_size: the number of crops along one dimension of the image grid.
    - crop_size: the size of one crop along one dimension (crops are square).
    """
    # Ensure that crops is a numpy array
    crops = crops.cpu().detach().numpy() if isinstance(crops, torch.Tensor) else np.array(crops)

    # Get the number of channels (should be 1 for grayscale, 3 for RGB)
    n_channels = crops.shape[1]

    # Initialize an empty array to hold the final stitched image
    full_img_size = grid_size * crop_size
    stitched_img = np.zeros((full_img_size, full_img_size, n_channels))

    # Reshape crops if necessary
    crops = crops.reshape((grid_size, grid_size, crop_size, crop_size, n_channels))

    # Iterate over each crop and place it in the correct position in the stitched image
    for i in range(grid_size):
        for j in range(grid_size):
            top = i * crop_size
            left = j * crop_size

            stitched_img[top:top+crop_size, left:left+crop_size] = crops[i, j]

    return stitched_img

# %% [cell 56]
import os
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

def generate_and_save_images(generator, dataloader, n_images, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the dataloader and generate predictions
    for i, batch in enumerate(dataloader):
        # Stop after generating n_images
        if i >= n_images:
            break

        # Convert the batch list into a tensor
            
        # Generate predictions
        batch = batch[0].to(device)
        predictions = generator(batch)
        
        # print(predictions.shape)


        # Stitch the predictions
        stitched_prediction = stitch_crops(predictions, grid_size=8, crop_size=256)

        # Stitch the original image
        stitched_original = stitch_crops(batch, grid_size=8, crop_size=256)

        # Convert to uint16 and scale
        stitched_original_uint16 = (stitched_original * max_trans).astype(np.uint16)
        stitched_prediction_uint16 = (stitched_prediction * max_fluo).astype(np.uint16)

        # Save the original and the prediction as TIFF
        io.imsave(os.path.join(output_dir, f'original_{i}.tif'), stitched_original_uint16)
        io.imsave(os.path.join(output_dir, f'prediction_{i}.tif'), stitched_prediction_uint16)

# Usage example:
generate_and_save_images(generator, val_loader, 10, 'stitched_result_tritc')

# %% [cell 57]
next(iter(val_loader))[0].shape

# %% [cell 58]
example_input = torch.rand(64, 1, 256, 256).to(device)

traced_script_module = torch.jit.trace(generator, example_input)

os.makedirs("jit-models", exist_ok=True)

torch.jit.save(traced_script_module, "tritc-xsp-wo-pretrain.pt")

# %% [cell 59]
# !mv tritc-xsp-wo-pretrain.pt jit-models/

# %% [cell 60]
# !ls jit-models/

# %% [cell 61]
loaded_model = torch.jit.load("jit-models/tritc-xsp-wo-pretrain.pt")

# %% [cell 62]
loaded_model.eval()

# %% [cell 63]
loaded_model(example_input)

# %% [cell 64]

# %% [cell 65]

# %% [cell 66]

# %% [cell 67]
predictions[0].shape

stitched_img = stitch_crops(predictions[0], grid_size=8, crop_size=256)

original_img = next(iter(val_loader))

original_img[0].shape

original_img_stitched = stitch_crops(original_img[0], grid_size=8, crop_size=256)

fig = plt.figure(dpi=300)
plt.imshow(original_img_stitched)

stitched_img.shape

fig = plt.figure(dpi=300)
plt.imshow(stitched_img)

import numpy as np
from skimage import io

original_img_stitched_f = (original_img_stitched * 27169.22249999999).astype(np.uint16)
# Multiply by 864 and convert to uint16
stitched_img_f = (stitched_img * 864).astype(np.uint16)

os.makedirs("stitched_result", exist_ok=True)


# Save as TIFF
io.imsave('stitched_result/stitched_image_trans.tif', original_img_stitched_f)
io.imsave('stitched_result/stitched_image_fluoro.tif', stitched_img_f)

# %% [cell 68]
print(len(dataset_val))

# %% [cell 69]
visualize_results(val_dataset, generator, [20, 84, 36, 48], viz_dir_val)

# %% [cell 70]
# Use the function after training, with the index argument set to [20, 84, 36, 48]
visualize_results(dataset, generator, [20, 84, 36, 48], viz_dir_test)
clear_output(wait=True)

# %% [cell 71]
import os
import torch

# Specify your directories
checkpoint_dir = 'checkpoints/pix2pix_best_focus_v1/'
data_dir = 'dataset/pix2pix_best_focus_v1'

# Make sure the directories exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Training code and model definition would be here

# Saving the models
torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'generator.pth'))
torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))

# Saving the optimizers
torch.save(optimizer_G.state_dict(), os.path.join(checkpoint_dir, 'optimizer_G.pth'))
torch.save(optimizer_D.state_dict(), os.path.join(checkpoint_dir, 'optimizer_D.pth'))

# To load the models and optimizers, you'd need to define them first, then load the state:

generator = Generator(in_channels=1, out_channels=1)
generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'generator.pth')))

discriminator = Discriminator(in_channels=1)
discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'discriminator.pth')))

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer_G.pth')))

optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer_D.pth')))

# Save dataset paths
with open(os.path.join(data_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset.file_list, f)

# # To load the dataset
# with open(os.path.join(data_dir, 'dataset.json'), 'r') as f:
#     file_list = json.load(f)


# %% [cell 72]

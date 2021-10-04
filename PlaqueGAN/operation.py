# Adapted from code of https://github.com/odegeasslbc/FastGAN-pytorch
# with a few extras such as a training logger, HDF5 datasets etc.

import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import shutil
import json
import pandas as pd
import csv

# class for recording training information
class GANLogger(object):
    def __init__(self, file_dir, reinitialize=False, filename='train_log.csv'):
        self.root = file_dir
        self.filename = filename
        self.headers = ['iter','G_loss','D_loss_total','D_loss_real','D_loss_fake',
        'D_loss_aux','D_loss_LC','D_real','D_fake']
        # delete log if re-starting and log already exists
        self.file_path = os.path.join(self.root, self.filename)
        if os.path.exists(self.file_path):
            if reinitialize:
                os.remove(self.file_path)
                self.reinit(self.filename)
        else:
            self.reinit(self.filename)

    def reinit(self, filename):

        with open(self.file_path,'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.headers)
            writer.writeheader()

    def log(self, metrics):
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.headers)
            writer.writerow(metrics)

def get_config(csv_path, config_name, type='train'):
    if type == 'model':
        all_configs = pd.read_csv(csv_path, converters={'g_skip_map': eval,'g_skip_conn': eval, 'g_attn_layers': eval, 'd_skip_map': eval,'d_skip_conn': eval, 'd_attn_layers': eval})
    elif type == 'train':
        all_configs = pd.read_csv(csv_path, converters={'policy': eval})

    config_selected = all_configs[all_configs['identifier'] == config_name]
    config_dict = config_selected.to_dict(orient='records')[0]
    return config_dict

def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def get_dir(args):
    task_name = 'train_results/' + args.name
    saved_model_folder = os.path.join( task_name, 'models')
    saved_image_folder = os.path.join( task_name, 'images')
    fid_folder = os.path.join( task_name, 'fid')

    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)
    os.makedirs(fid_folder, exist_ok=True)

    for f in os.listdir('./'):
        if '.py' in f:
            shutil.copy(f, task_name+'/'+f)

    with open( os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder, fid_folder


class  ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None):
        super( ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img



from io import BytesIO
# import lmdb
#from torch.utils.data import Dataset
import h5py
"""
class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            #key_asp = f'aspect_ratio-{str(index).zfill(5)}'.encode('utf-8')
            #aspect_ratio = float(txn.get(key_asp).decode())

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
"""
class H5Dataset(Dataset):
    """Represents a HDF5 dataset.

    Params
    ------
    file_path: Path to directory containing the dataset
    recursive: If True, searches for h5/hdf5 files within subdirectories
    load_data: If True, loads all data at once into RAM. Otherwise, load lazily
    transform: PyTorch transforms to apply to every instance (default = None)

    """
    def __init__(self,
                file_path,
                transform = None):

        super(H5Dataset, self).__init__()

        self.path = file_path
        self.transform = transform
        with h5py.File(self.path, 'r') as h5_file:
            self.n_images, self.width, self.height, self.n_channels = h5_file['images'].shape

    def open_file(self, file_path):
        self.h5_file = h5py.File(self.path, 'r')

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.open_file(self.path)

        img = self.h5_file['images'][idx]

        if self.transform is not None:
            img = self.transform(img)

        return img

class MultilabelDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data_info = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        c=torch.Tensor(self.data_info.loc[:,'cored'])
        d=torch.Tensor(self.data_info.loc[:,'diffuse'])
        a=torch.Tensor(self.data_info.loc[:,'CAA'])
        c=c.view(c.shape[0],1)
        d=d.view(d.shape[0],1)
        a=a.view(a.shape[0],1)
        self.raw_labels = torch.cat([c,d,a], dim=1)
        self.labels = (torch.cat([c,d,a], dim=1)>0.99).type(torch.FloatTensor)

    def __getitem__(self, index):
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]
        raw_label = self.raw_labels[index]
        # Get image name from the pandas df
        single_image_name = str(self.data_info.loc[index,'imagename'])
        # Open image
        try:
            img_as_img = Image.open(self.img_path + single_image_name)
        except:
            single_image_name = single_image_name.split('/')[0] + '/neg_' + single_image_name.split('/')[1]
            img_as_img = Image.open(self.img_path + single_image_name)
        # Transform image to tensor
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)
        # Return image and the label
        return (img_as_img, single_image_label, raw_label, single_image_name)

    def __len__(self):
        return len(self.data_info.index)

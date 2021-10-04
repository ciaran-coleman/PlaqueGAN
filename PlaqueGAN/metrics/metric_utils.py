import torch
from torch.cuda import amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from scipy.stats import truncnorm
import pandas as pd
import numpy as np

from tqdm import tqdm

import os
import copy

from torch.nn.functional import adaptive_avg_pool2d

from metrics.vgg16 import VGG16Features
from pytorch_fid.inception import InceptionV3

def get_feature_detector(model_type = 'vgg16', ckpt_dir = './metric_classifiers',
                            load_ckpt = False, pre_train = True, fc_dim = 4096,
                            device = torch.device('cpu')):
    if model_type == 'vgg16':
        feature_detector = VGG16Features(device=device, pre_train=pre_train, fc_dim=fc_dim)

        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir,f'vgg16_R{fc_dim}.pth')
            ckpt = torch.load(ckpt_path)
            feature_detector.load_state_dict(ckpt['state_dict'])

    elif model_type == 'inception':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[fc_dim]
        feature_detector = InceptionV3([block_idx])

    return feature_detector.eval().to(device)

@torch.no_grad()
def get_features_dataset(dataloader, detector_type='vgg16', pre_train=True, load_ckpt = False,
                        fc_dim=4096, device = torch.device('cpu')):

    # initialise
    feature_detector = get_feature_detector(model_type=detector_type, load_ckpt=load_ckpt,
                            pre_train=pre_train, fc_dim=fc_dim, device=device)

    features_all = []
    for images in tqdm(dataloader, total=len(dataloader),desc='processing real images'):
        if type(images) == list:
            images = images[0]
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        images = images.to(device)
        if detector_type=='inception':
            features = feature_detector(images)[0]
            if features.size(2) != 1 or features.size(3) != 1:
                features = adaptive_avg_pool2d(features, output_size=(1,1))

            features = features.squeeze(3).squeeze(2).cpu().numpy()
            # features = feature_detector(images)[0].squeeze(3).squeeze(2).cpu().numpy()
        else:
            features = feature_detector(images).cpu().numpy()
        features_all.append(features)

    features_all = np.concatenate(features_all, axis=0)

    return features_all

@torch.no_grad()
def get_features_generator(netG, detector_type='vgg16', pre_train=True, load_ckpt = False,
                        fc_dim=4096, num_samples = 10000, z_dim = 256, batch_size = 16,
                        batch_gen = 8, trunc = 0, device = torch.device('cpu')):
    netG = copy.deepcopy(netG).requires_grad_(False).to(device)
    # netG.to(device)
    # generate images
    def run_generator(netG, z, norm=False):
        # with amp.autocast():
        imgs = netG(z)[0]
        # convert images from -1 1 to 0 255 uint8 (as would be done if saving)
        imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # convert back to torch FloatTensor
        imgs = imgs.to(torch.float32).div_(255)
        # normalize
        if norm:
            # using ImageNet mean and std
            imgs = Batched_Normalize(imgs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return imgs

    # initialise
    feature_detector = get_feature_detector(model_type=detector_type, load_ckpt=load_ckpt,
                            pre_train=pre_train, fc_dim=fc_dim, device=device)
    # main loop
    features_all =[]
    num_loops = -(-num_samples // batch_size) #ceiling division
    for i in tqdm(range(num_loops), total=num_loops, desc='processing generated images'):
    # while not is_full(features_all, num_samples):
        images = []
        for j in range(batch_size // batch_gen):
            z = truncated_z_sample(batch_gen, z_dim, truncation=trunc).to(device)
            # print(torch.max(z))
            # z = torch.randn(batch_gen, z_dim).to(device)
            images.append(run_generator(netG, z, norm=False))
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1,3,1,1])


        if detector_type=='inception':
            features = feature_detector(images)[0]
            if features.size(2) != 1 or features.size(3) != 1:
                features = adaptive_avg_pool2d(features, output_size=(1,1))

            features = features.squeeze(3).squeeze(2).cpu().numpy()

            # features = feature_detector(images)[0].squeeze(3).squeeze(2).cpu().numpy()
        else:
            features = feature_detector(images).cpu().numpy()
        # features = feature_detector(images).cpu().numpy()
        features_all.append(features)

    features_all = np.concatenate(features_all, axis=0)

    return features_all[:num_samples, :]

def Batched_Normalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype = tensor.dtype, device=tensor.device)[None, :, None, None]
    std = torch.as_tensor(std,  dtype = tensor.dtype, device=tensor.device)[None, :, None, None]

    tensor = tensor.sub_(mean).div_(std)

    return tensor

#https://github.com/ajbrock/BigGAN-PyTorch/
def truncated_z_sample(batch_size, z_dim, truncation = 0.5, seed = None):
    state = None if seed is None else np.random.RandomState(seed)
    if truncation > 0:
        values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
        return torch.as_tensor(truncation * values, dtype=torch.float32)
    else:
        return torch.randn((batch_size, z_dim))

def accumulate_standing_stats(netG, batch_size, z_dim, num_accumulations=100):
    netG.train()
    for i in range(num_accumulations):
        z = torch.randn(batch_size, z_dim, requires_grad = False)
        _ = netG(z)

    # switch back to evaluation mode
    netG.eval()

def save_prdc_nn_score(scores, save_dir, embedding_type, n_fakes):
    # create dataframe for saving
    df_score = pd.DataFrame(data=scores, columns=['iterations','P','R','D','C', 'acc', 'acc_r','acc_f', 'nn_P', 'nn_R'])

    df_score.to_csv(os.path.join(save_dir, f'prdc_nn_{embedding_type}_{n_fakes}.csv'))

def save_kid_score(scores, save_dir, feature_size, n_fakes):
    # create dataframe for saving
    df_score = pd.DataFrame(data=scores, columns=['iterations','kid','kid_std'])

    df_score.to_csv(os.path.join(save_dir, f'kid_f{feature_size}_{n_fakes}.csv'), index=False)

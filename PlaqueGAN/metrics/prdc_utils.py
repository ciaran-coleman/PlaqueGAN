import torch
from torch.cuda import amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

from tqdm import tqdm

import os
import copy

# from metrics.vgg16 import VGG16Features
# from pytorch_fid.inception import InceptionV3
from metrics.metric_utils import Batched_Normalize, truncated_z_sample, get_feature_detector

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# def get_feature_detector(model_type = 'vgg16', ckpt_dir = './metric_classifiers',
#                             load_ckpt = False, pre_train = True, fc_dim = 4096,
#                             device = torch.device('cpu')):
#     if model_type == 'vgg16':
#         feature_detector = VGG16Features(device=device, pre_train=pre_train, fc_dim=fc_dim)
#
#         if load_ckpt:
#             ckpt_path = os.path.join(ckpt_dir,f'vgg16_R{fc_dim}.pth')
#             ckpt = torch.load(ckpt_path)
#             feature_detector.load_state_dict(ckpt['state_dict'])
#
#     elif model_type == 'inception':
#         block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[fc_dim]
#         feature_detector = InceptionV3([block_idx])
#
#     return feature_detector.eval().to(device)

def is_full(x, max_samples):
    return len(x) >= max_samples

# @torch.no_grad()
# def get_features_dataset(dataloader, detector_type='vgg16', pre_train=True, load_ckpt = False,
#                         fc_dim=4096, device = torch.device('cpu')):
#
#     # initialise
#     feature_detector = get_feature_detector(model_type=detector_type, load_ckpt=load_ckpt,
#                             pre_train=pre_train, fc_dim=fc_dim, device=device)
#
#     features_all = []
#     for images in tqdm(dataloader, total=len(dataloader),desc='processing real images'):
#         if images.shape[1] == 1:
#             images = images.repeat([1, 3, 1, 1])
#
#         images = images.to(device)
#         features = feature_detector(images).cpu().numpy()
#         features_all.append(features)
#
#     features_all = np.concatenate(features_all, axis=0)
#
#     return features_all
#
# @torch.no_grad()
# def get_features_generator(netG, detector_type='vgg16', pre_train=True, load_ckpt = False,
#                         fc_dim=4096, num_samples = 10000, z_dim = 256, batch_size = 16,
#                         batch_gen = 8, trunc = 0, device = torch.device('cpu')):
#     netG = copy.deepcopy(netG).requires_grad_(False).to(device)
#
#     # generate images
#     def run_generator(netG, z, norm=False):
#         # with amp.autocast():
#         imgs = netG(z)[0]
#         # convert images from -1 1 to 0 255 uint8 (as would be done if saving)
#         imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#         # convert back to torch FloatTensor
#         imgs = imgs.to(torch.float32).div_(255)
#         # normalize
#         if norm:
#             # using ImageNet mean and std
#             imgs = Batched_Normalize(imgs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         return imgs
#
#     # initialise
#     feature_detector = get_feature_detector(model_type=detector_type, load_ckpt=load_ckpt,
#                             pre_train=pre_train, fc_dim=fc_dim, device=device)
#     # main loop
#     features_all =[]
#     num_loops = -(-num_samples // batch_size) #ceiling division
#     for i in tqdm(range(num_loops), total=num_loops, desc='processing generated images'):
#     # while not is_full(features_all, num_samples):
#         images = []
#         for j in range(batch_size // batch_gen):
#             z = truncated_z_sample(batch_gen, z_dim, truncation=trunc).to(device)
#             # print(torch.max(z))
#             # z = torch.randn(batch_gen, z_dim).to(device)
#             images.append(run_generator(netG, z, norm=pre_train))
#         images = torch.cat(images)
#         if images.shape[1] == 1:
#             images = images.repeat([1,3,1,1])
#
#
#         features = feature_detector(images).cpu().numpy()
#         features_all.append(features)
#
#     features_all = np.concatenate(features_all, axis=0)
#
#     return features_all[:num_samples, :]

def save_prdc_score(iteration, prdc, prdc_dir, embedding_type, n_fakes):
    # prdc passed in as directory. separate
    p = prdc['precision']
    r = prdc['recall']
    d = prdc['density']
    c = prdc['coverage']
    with open(os.path.join(prdc_dir, f'prdc_scores_{n_fakes}_{embedding_type}.txt'), 'a') as f:
        f.write(f'{iteration},{p}, {r}, {d}, {c}\n')

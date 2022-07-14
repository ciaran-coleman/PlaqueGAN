# code for calculating KID for a single PlaqueGAN experiment - calculates
# KID on all checkpoints of the model saved during training.

import os
from shutil import rmtree
from pathlib import Path
from random import random

import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.utils as vutils
from torchvision import transforms

import argparse
from tqdm import tqdm

import numpy as np
from glob import glob

from operation import load_params, H5Dataset, get_config
from numpy_transforms import NumpyToTensor
from models import Generator
from metrics.kid import compute_kid
from metrics.metric_utils import get_features_generator, get_features_dataset, save_kid_score

# get the inception model
from pytorch_fid.inception import InceptionV3
import json

torch.backends.cudnn.benchmark = True
@torch.no_grad()
def reprocess_kid(args):
    """
    Wrapper to post-process/reprocess precision/recall metrics across an experiment for checkpoints
    """

    base_dir = args.path
    data_path = args.data_path
    exp_name = args.name
    start_iter = args.start_iter
    end_iter = args.end_iter
    batch_size = args.batch_size
    batch_gen = args.batch_gen
    n_fakes = args.n_fakes
    trunc = args.trunc
    n_subsets = args.n_subsets
    subset_size = args.subset_size
    clear_metrics = args.clear_metrics
    im_size = args.im_size
    ema = args.ema
    use_temp = args.use_temp_folder
    feature_size = args.feature_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader_workers = 2

    # noise_dim = 256
    if use_temp:
        save_dir = './temp/evaluation/metrics'
        clear_metrics = 1
    else:
        save_dir = os.path.join(base_dir, exp_name, 'evaluation','metrics')

    if clear_metrics == 1 and os.path.exists(save_dir):
        rmtree(save_dir, ignore_errors=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # list of valid checkpoints
    ckpts = glob(os.path.join(base_dir, exp_name, 'models', 'all_*.pth'))
    ckpts = sorted(list(map(os.path.basename, ckpts)), key=lambda x: int(x.partition('_')[-1].partition('.')[0]))
    ckpts = [x for x in ckpts if (int(x.partition('_')[-1].partition('.')[0]) <= end_iter) and (int(x.partition('_')[-1].partition('.')[0]) >= start_iter)]

    # create generator using the arguments used to run experiments
    with open(os.path.join(base_dir, exp_name, 'args.txt'), mode='r') as f:
        args_train = json.load(f)
        model_config = args_train['model_config']
        model_config = get_config('model_configs.csv', model_config, type='model')
        # model_config = get_config('model_configs_trial.csv', model_config, type='model')
        noise_dim = model_config['nz']

        netG = Generator(
                nz                  = model_config['nz'],
                activation          = model_config['g_activation'],
                chan_attn           = model_config['g_chan_attn'],
                sle_map             = model_config['g_skip_map'],
                skip_conn           = model_config['g_skip_conn'],
                spatial_attn        = model_config['g_spatial_attn'],
                attn_layers         = model_config['g_attn_layers'],
                conv_layers         = model_config['g_conv_layers'],
                alternate_layers    = model_config['g_alternate_layers'],
                anti_alias          = model_config['g_anti_alias'],
                noise_inj           = model_config['g_noise_inj'],
                multi_res_out       = model_config['g_multi_res_out'],
                small_im_size       = model_config['g_small_im_size'],
                use_tanh            = model_config['use_tanh']
        )

        print('loaded!')

    if os.path.splitext(data_path)[1] == '.npy':
        # real features can be loaded in without needing to process
        process_real = False
        real_features = np.load(data_path)
    elif os.path.splitext(data_path)[1] == '.h5':
        process_real = True
        # create the DataLoader
        trans = transforms.Compose([NumpyToTensor()])

    kid_dataset = H5Dataset(file_path=data_path, transform=trans)
    if n_fakes == 0:
        n_fakes = len(kid_dataset)
        print(n_fakes)

    kid_dataloader = DataLoader(kid_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=dataloader_workers)

    # process real features if needed - only need to do once
    if process_real:
        real_features = get_features_dataset(kid_dataloader, detector_type='inception',
                                            fc_dim=feature_size, device=device)

    scores = np.zeros((len(ckpts),3))

    for i,ckpt in tqdm(enumerate(ckpts), total=len(ckpts), desc='Processing checkpoints'):

        iteration = int(ckpt.partition('_')[-1].partition('.')[0])
        scores[i,0] = iteration
        ckpt_path = os.path.join(base_dir,exp_name,'models',ckpt)
        checkpoint = torch.load(ckpt_path)

        if ema:
            load_params(netG, checkpoint['g_ema'])
        else:
            netG.load_state_dict(checkpoint['g'])

        # process generator features
        gen_features = get_features_generator(netG, num_samples=n_fakes, z_dim=noise_dim,
                                            fc_dim=feature_size, detector_type='inception',
                                            batch_size=batch_size, batch_gen=batch_gen,
                                            trunc = trunc, device=device)

        kid = compute_kid(real_features, gen_features, n_subsets = n_subsets, max_subset_size = subset_size)
        scores[i,1] = kid['mean']
        scores[i,2] = kid['std']

    # save scores
    save_kid_score(scores, save_dir, feature_size, n_fakes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gan_kid')

    parser.add_argument('--path', type=str, default='./train_results', help='Base directory where train results are stored')
    parser.add_argument('--data_path', type=str, default= './train_data/cored_train.h5')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--im_size', type=int, default=256, help='size of image')
    parser.add_argument('--start_iter', type=int, default=5000, help='the iteration to start fid calculation')
    parser.add_argument('--end_iter', type=int, default=50000, help='the iteration to stop fid calculation')
    parser.add_argument('--n_fakes', type=int, default=0, help='number of fake images to use for kid calculation. If 0, will match total number of reals.')
    parser.add_argument('--batch_gen', type=int, default=8, help='how many images to generate at a time (limited by GPU memory)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size passed through feature extractor (limited by GPU memory)')
    parser.add_argument('--n_subsets', type=int, default=10, help='number of times to calculate MMD')
    parser.add_argument('--subset_size', type=int, default=1000, help='subset size to calculate MMD')
    parser.add_argument('--ema', type=int, default=1, help='boolean flag where 1= generate images using EMA and 0 is default.')
    parser.add_argument('--feature_size', type=int, default=2048, help='Feature size to use in inception network.')
    parser.add_argument('--trunc', type=float, default=0, help='truncation threshold to apply when sampling latent z. If 0, no truncation applied.')
    parser.add_argument('--clear_metrics', type=int, default=0, help='boolean flag where 1= clear entire metric folder and 0=keep.')
    parser.add_argument('--use_temp_folder', type=int, default=1, help='Elect to use temporary folder.')

    args = parser.parse_args()
    print(args)

    reprocess_kid(args)

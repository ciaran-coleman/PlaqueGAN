import os
import json
import shutil
import torch
import numpy as np
import random

import torchvision
from torchvision import transforms

import torch.nn as nn
from custom_losses import FocalLoss, AsymmetricLoss

def get_dir(args):
    experiment_name = "U_" + args.upsample_type + "_A_" + args.aug_type + "_L_" + args.loss_fn
    if args.save_hdd:
        experiment_dir = os.path.join("D:/ucl_masters_data/project/augmentation_experiments", experiment_name)
    else:
        experiment_dir = "train_results/" + experiment_name
    best_model_dir = os.path.join(experiment_dir, "models", "best")
    ckpt_model_dir = os.path.join(experiment_dir, "models", "ckpts")
    results_dir = os.path.join(experiment_dir, "results")

    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(ckpt_model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # save the arguments used
    with open(os.path.join(experiment_dir,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return best_model_dir, ckpt_model_dir, results_dir

def get_csv_paths(upsample_type):
    csv_paths = dict()
    if upsample_type == 'none':
        csv_path = './CSVs/train.csv'
    elif upsample_type == 'simple':
        csv_path = './CSVs/train_simple_up.csv'
    elif upsample_type == 'stdaug':
        csv_path = './CSVs/train_stdaug_up.csv'
    elif upsample_type == 'augmix':
        csv_path = './CSVs/train_augmix_up.csv'
    elif upsample_type == 'gan':
        csv_path = './CSVs/train_gan_up.csv'
    elif upsample_type == 'smote':
        csv_path = './CSVs/train_smote_up.csv'

    csv_paths['train'] = csv_path
    csv_paths['dev'] = './CSVs/dev.csv'
    csv_paths['test'] = './CSVs/test.csv'
    csv_paths['test_alt'] = './CSVs/test_alt.csv'
    csv_paths['test_combined'] = './CSVs/test_combined.csv'
    return csv_paths

def get_data_dirs():
    data_dirs = dict()
    # data_dirs['train'] = 'D:/ucl_masters_data/project/data/tiles/train_and_val/'
    # data_dirs['dev'] = 'D:/ucl_masters_data/project/data/tiles/train_and_val/'
    # data_dirs['test'] = 'D:/ucl_masters_data/project/data/tiles/test/'
    data_dirs['train'] = '../Plaquebox/plaquebox-paper-master/data/tiles/train_and_val/'
    data_dirs['dev'] = '../Plaquebox/plaquebox-paper-master/data/tiles/train_and_val/'
    data_dirs['test'] = '../Plaquebox/plaquebox-paper-master/data/tiles/hold-out/'
    return data_dirs
# set the random seeds for reproducibility
def set_seed(repeat):
    seeds = [123456789, 42, 111]
    seed_use = seeds[repeat]

    random.seed(seed_use)
    np.random.seed(seed_use)
    torch.cuda.manual_seed(seed_use)



# def write_results(results_dir, file_name, epoch, metric):

def get_transformations(norm, aug_type):
    data_transforms = dict()

    if aug_type == 'none':
        data_transforms['train'] = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm['mean'], norm['std'])
                                    ])
    if aug_type == 'standard':
        # Plaquebox augmentations
        data_transforms['train'] = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomRotation(180),
                            transforms.ColorJitter(brightness=0.1, contrast=0.2,saturation=0.2, hue=0.02),
                            transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.9,1.1), shear=10),
                            transforms.ToTensor(),
                            transforms.Normalize(norm['mean'], norm['std'])
                            ])

    elif aug_type == 'mixup':

        # use a reduced set of standard transformations
        data_transforms['train'] = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomResizedCrop(size=(256,256)),
                            transforms.ToTensor(),
                            transforms.Normalize(norm['mean'], norm['std'])
                            ])

    elif aug_type == 'mixup_alt':

        # use a reduced set of standard transformations
        data_transforms['train'] = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomResizedCrop(size=(256,256)),
                            transforms.ToTensor(),
                            transforms.Normalize(norm['mean'], norm['std'])
                            ])

    elif aug_type == 'samplepairing':
        data_transforms['train'] = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(norm['mean'], norm['std'])
                            ])
    elif aug_type == 'gan':
        # no augmentations applied. Augmentations come in the fact that there are additional
        # synthetic samples.
        data_transforms['train'] = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(norm['mean'], norm['std'])
                            ])

    data_transforms['dev'] = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(norm['mean'], norm['std'])
                        ])

    data_transforms['test'] = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(norm['mean'], norm['std'])
                        ])

    return data_transforms

def get_loss_fn(loss_fn, weights = None):
    if loss_fn == 'simple':
        criterion = nn.MultiLabelSoftMarginLoss(reduction='sum')

    elif loss_fn == 'weighted':
        criterion = nn.MultiLabelSoftMarginLoss(weights=weights, reduction='sum')

    elif loss_fn == 'focal':
        criterion = FocalLoss(gamma = 2.0, alpha = 0.25)

    elif loss_fn == 'asym':
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    return criterion

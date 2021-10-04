# Main script for running augmentation experiments

import time, os
import glob
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from torch.cuda import amp
import sys
import torchvision.utils as vutils

import argparse
import json
from tqdm import tqdm

from models import Net, Net_Faster
from datasets import MultilabelDataset, ReducedMultilabelDataset
from operations import *
from eval_metrics import *

from augmentation_utils import mixup_data, construct_mixup_args

torch.backends.cudnn.benchmark=True

# separating training run and validation run
def fit(model, epoch, dataloader, dataset_size, metrics_dict, mixup_args, optimizer,
criterion, scheduler, scaler = None, device= torch.device('cpu')):
    # set model to training mode
    model.train()
    running_loss = 0.0
    running_corrects = torch.zeros(model.num_classes).to(device)
    running_preds = torch.Tensor(0).to(device)
    running_labels = torch.Tensor(0)

    for i, data in tqdm(enumerate(dataloader), total = len(dataloader)):
        # zero the parameter gradients
        optimizer.zero_grad()

        # get inputs and move to specified device
        inputs, labels, raw_labels, names = data
        running_labels = torch.cat([running_labels, labels])
        inputs, labels = inputs.to(device), labels.to(device)

        if mixup_args['mix']: # apply mixup
            inputs, labels = mixup_data(inputs, labels, mixup_args['alpha'], mixup_args['label_construct'], device=device)

        # forward pass
        if scaler is not None: # using AMP
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        preds = torch.sigmoid(outputs)
        predictions = (preds>0.5).long()
        running_preds = torch.cat([running_preds, preds.detach()])
        running_loss += loss.item()
        running_corrects += torch.sum(predictions==labels, 0)

    scheduler.step() # step each epoch (based on what is done in Plaquebox)

    running_preds = running_preds.to('cpu')
    running_corrects = running_corrects.to('cpu')
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    # calculate precision and recall
    pr_auc_macro, pr_auc_macro_list = macro_avg_auc(running_preds, running_labels, type='prc')
    pr_auc_micro = micro_avg_auc(running_preds, running_labels, type='prc')

    model.train_loss_curve.append(epoch_loss)
    model.train_aucpr_macro_curve.append(pr_auc_macro)
    model.train_aucpr_micro_curve.append(pr_auc_micro)

    # update metrics
    update_metrics_dict(metrics_dict, 'train', epoch, epoch_loss, epoch_acc.numpy(), pr_auc_micro, pr_auc_macro ,pr_auc_macro_list)
    print('{} Loss: {:.4f}\n Cored: {:.4f} Diffuse: {:.4f} CAA: {:.4f}'.format(
                'train', epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2]))
    print('{} AUC Micro: {:.4f}\n Cored: {:.4f} Diffuse: {:.4f} CAA: {:.4f}'.format(
                'train', pr_auc_micro, pr_auc_macro_list[0], pr_auc_macro_list[1], pr_auc_macro_list[2]))

    return metrics_dict

def validate(model, repeat, epoch, dataloader, dataset_size, metrics_dict, criterion, scaler,
best_model, best_pr_auc, best_loss, best_model_dir, patience, device= torch.device('cpu')):
    # set model to eval mode
    model.eval()
    running_loss = 0.0
    running_corrects = torch.zeros(model.num_classes).to(device)
    running_preds = torch.Tensor(0).to(device)
    running_labels = torch.Tensor(0)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total = len(dataloader)):
            # get inputs and move to specified device
            inputs, labels, raw_labels, names = data
            running_labels = torch.cat([running_labels, labels])
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            if scaler is not None: # using AMP
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = torch.sigmoid(outputs)
            predictions = (preds>0.5).long()
            running_preds = torch.cat([running_preds, preds])

            running_loss += loss.item()
            running_corrects += torch.sum(predictions==labels, 0)

    running_preds = running_preds.to('cpu')
    running_corrects = running_corrects.to('cpu')
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    # calculate precision and recall
    pr_auc_macro, pr_auc_macro_list = macro_avg_auc(running_preds, running_labels, type='prc')
    pr_auc_micro = micro_avg_auc(running_preds, running_labels, type='prc')

    model.dev_loss_curve.append(epoch_loss)
    model.dev_aucpr_macro_curve.append(pr_auc_macro)
    model.dev_aucpr_micro_curve.append(pr_auc_micro)

    # update metrics
    update_metrics_dict(metrics_dict, 'dev', epoch, epoch_loss, epoch_acc.numpy(), pr_auc_micro, pr_auc_macro ,pr_auc_macro_list)

    # compare to best_model
    patience += 1

    if pr_auc_macro > best_pr_auc:
        best_pr_auc = pr_auc_macro
        best_model = copy.deepcopy(model).to('cpu')
        # save this new best model
        model_name = f"{repeat}_prauc"
        path_save = best_model_dir +'/' + model_name + '.pt'
        torch.save({'model_state_dict': model.state_dict()}, path_save)
        print('best pr_auc: ', best_pr_auc)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model = copy.deepcopy(model).to('cpu')
        model_name = f"{repeat}_loss"
        path_save = best_model_dir +'/' + model_name + '.pt'
        torch.save({'model_state_dict': model.state_dict()}, path_save)
        print('best loss: ', best_loss)
        patience = 0 # reset patience if best loss achieved

    print('{} Loss: {:.4f}\n Cored: {:.4f} Diffuse: {:.4f} CAA: {:.4f}'.format(
                'dev', epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2]))
    print('{} AUC Micro: {:.4f}\n Cored: {:.4f} Diffuse: {:.4f} CAA: {:.4f}'.format(
                'dev', pr_auc_micro, pr_auc_macro_list[0], pr_auc_macro_list[1], pr_auc_macro_list[2]))

    return best_model, best_pr_auc, best_loss, metrics_dict, patience

def run_experiment(args):
    upsample_type = args.upsample_type
    aug_type = args.aug_type
    early_stop = args.early_stop
    n_repeats = [args.n_repeats] if type(args.n_repeats)!=list else args.n_repeats
    n_epochs = args.n_epochs
    loss_fn = args.loss_fn
    lr = args.lr
    wd = args.wd
    lr_step_size = args.lr_step_size
    lr_gamma = args.lr_gamma
    batch_size = args.batch_size
    with_amp = args.with_amp
    load_ckpt = args.load_ckpt

    dataloader_workers = 2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # construct mixup args if needed
    mixup_args = construct_mixup_args(aug_type)
    # get the directories to save models and results in
    best_model_dir, ckpt_model_dir, results_dir = get_dir(args)

    # set which csv to use for dataset based on upsampling strategy
    csv_paths = get_csv_paths(upsample_type)
    print(csv_paths)
    # load color normalization info
    norm = np.load('./utils/normalization.npy', allow_pickle = True).item()

    # get the data transformations to apply
    data_transforms = get_transformations(norm, aug_type)
    # location of the data
    data_dirs = get_data_dirs()

    # setup the dataset and dataloaders
    image_datasets = {x: MultilabelDataset(csv_paths[x], data_dirs[x], data_transforms[x])
                  for x in ['train', 'dev']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'dev']}
    print("{} training images, {} validation images".format(dataset_sizes["train"], dataset_sizes["dev"]))

    for repeat in n_repeats:
        since = time.time()
        # initialise dictionary of metrics
        metrics_dict = init_metrics_dict()
        # set the manual seed for the experiment. Each repeat will use a different seed.
        set_seed(repeat)

        # name to save model ckpt
        save_name = 'ckpt_latest'
        # initialise dataloaders
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=batch_size,
                                                        shuffle = True,
                                                        num_workers = dataloader_workers,
                                                        pin_memory=True) for x in ['train','dev']}

        # initialise model
        model = Net()
        model.to(device)
        torch.backends.cudnn.benchmark=True
        # define optimiser and criterion for loss
        if loss_fn == 'weighted':
            # update this
            weights = torch.FloatTensor([1,1,1])
            weights = weight.to(device)
        else:
            weights = None

        criterion = get_loss_fn(loss_fn, weights=weights)
        # Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
        # learning rate scheduler - Plaquebox
        scheduler = lr_scheduler.StepLR(optimizer, step_size = lr_step_size, gamma = lr_gamma)

        # gradient scalar for AMP training
        if with_amp:
            scaler = amp.GradScaler()
        else:
            scaler = None

        # initialise metrics to track training progress
        best_loss = np.inf
        best_pr_auc = 0
        best_model = None
        patience = 0

        # overwrite if loading from checkpoint
        if load_ckpt:
            ckpt_path = ckpt_model_dir +'/' + save_name + '.pt'
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            last_epoch = ckpt['last_epoch']
            best_loss = ckpt['best_loss']
            best_pr_auc = ckpt['best_pr_auc']
            patience = ckpt['patience']
            metrics_dict = ckpt['metrics_dict']
            model.train_loss_curve = ckpt['train_loss_curve']
            model.dev_loss_curve = ckpt['dev_loss_curve']
            model.train_aucpr_macro_curve = ckpt['train_aucpr_macro_curve']
            model.train_aucpr_micro_curve = ckpt['train_aucpr_micro_curve']
            model.dev_aucpr_macro_curve = ckpt['dev_aucpr_macro_curve']
            model.dev_aucpr_micro_curve = ckpt['dev_aucpr_micro_curve']
            torch.random.set_rng_state(ckpt['torch_rand_state'])
            np.random.set_state(ckpt['np_rand_state'])

        for epoch in range(1,n_epochs+1):
            epoch_start = time.time()
            print('Epoch {}/{}'.format(epoch, n_epochs))
            print('-' * 10)

            # train
            metrics_dict = fit(model, epoch, dataloaders['train'], dataset_sizes['train'], metrics_dict, mixup_args,
                                optimizer, criterion, scheduler, scaler = scaler, device= device)

            # validate
            best_model, best_pr_auc, best_loss, metrics_dict, patience = validate(model, repeat, epoch, dataloaders['dev'], dataset_sizes['dev'],
                                                                metrics_dict, criterion, scaler, best_model, best_pr_auc, best_loss,
                                                                best_model_dir, patience, device= device)

            epoch_end = time.time() - epoch_start
            print('train, Epoch time {:.0f}m {:.0f}s'.format(
                    epoch_end // 60, epoch_end % 60))
            print()

            if (early_stop > 0) and (patience == early_stop):
                break

            # save checkpoint every epoch for easy resuming
            path_save = ckpt_model_dir +'/' + save_name + '.pt'
            torch.save({'last_epoch': epoch,
            'torch_rand_state': torch.random.get_rng_state(),
            'np_rand_state': np.random.get_state(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'patience': patience,
            'best_loss': best_loss,
            'best_pr_auc': best_pr_auc,
            'metrics_dict': metrics_dict,
            'train_loss_curve': model.train_loss_curve,
            'dev_loss_curve': model.dev_loss_curve,
            'train_aucpr_macro_curve':model.train_aucpr_macro_curve,
            'train_aucpr_micro_curve':model.train_aucpr_micro_curve,
            'dev_aucpr_macro_curve':model.dev_aucpr_macro_curve,
            'dev_aucpr_micro_curve':model.dev_aucpr_micro_curve,},
            path_save)

        # save metrics
        save_metrics_dict(results_dir, metrics_dict, repeat)
        print(f'Run {repeat} completed!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='aug trials')

    parser.add_argument('--upsample_type', type=str, default='simple', help='Determine upsampling strategy to apply. Alternatives = stdaug, augmix, smote or gan.')
    parser.add_argument('--aug_type', type=str, default='standard', help = 'Determine data augmentation strategy. Alternatives = none, mixup, mixup_alt, samplepairing, gan.')
    parser.add_argument('--early_stop', type=int, default=10, help='sets patience of early stopping. If early stopping = 0, it is not used')
    parser.add_argument('--n_repeats', nargs='+', type=int, default=0, help='Repeats to run. 0 1 2 runs all 3 repeats with respective random seeds.')
    parser.add_argument('--loss_fn', type=str, default='simple', help='which loss function to use. alternatives = weighted, focal or asym.')
    parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs to run for. Set to 60 to align with Plaquebox.')
    parser.add_argument('--lr', type=float, default=0.00008, help='learning rate.')
    parser.add_argument('--wd', type=float, default=0.008, help='weight decay.')
    parser.add_argument('--lr_step_size', type=int, default=15, help='number of epochs before decaying learning rate.')
    parser.add_argument('--lr_gamma', type=float, default=0.4, help='scaling factor for learning rate when decaying.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--with_amp', type=int, default=1, help='flag to determine whether to use Automatic Mixed Precision training.')
    parser.add_argument('--save_hdd', type=int, default=1, help='flag to save to HDD instead of SSD.')
    parser.add_argument('--load_ckpt', type=int, default=0, help='determine whether to continue training from checkpoint')
    args = parser.parse_args()
    print(args)
    run_experiment(args)

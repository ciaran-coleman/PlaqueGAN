# main code for training PlaqueGAN
# Adapted from the official implementation of the paper by Liu et al.:
# "Towards Faster and Stabilized GAN Training for High-Fidelity Few Shot Image Synthesis"
# https://github.com/odegeasslbc/FastGAN-pytorch

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir, get_config
from operation import ImageFolder, InfiniteSamplerWrapper, H5Dataset, GANLogger
from diffaug import DiffAugment
from numpy_transforms import NumpyRandomFlip, NumpyToTensor

from kornia import resize
import lpips
import losses

from torch.cuda import amp
from contextlib import contextmanager

torch.backends.cudnn.benchmark = True

@contextmanager
def null_context():
    yield

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def resize_fn(data, size_out, type = 'simple', trans = None):
    if type == 'simple':
        return F.interpolate(data, size_out)
    elif type == 'bilin':
        return F.interpolate(data, size_out, mode = 'bilinear')
    elif type == 'antialias':
        return resize(data, size_out, interpolation='bilinear', antialias=True)
    elif type == 'pil':
        img_out = torch.zeros((img.shape[0],img.shape[1], size_out, size_out), device = torch.device("cuda"))
        for i, img in enumerate(img):
            img_out[i] = trans(img.add(1).mul(0.5))
        return trans(data.add(1).mul(0.5))

def calc_aux_loss(data, part, rec_list, aux_loss_fn, resize_method, mse_scaling):
    if len(rec_list) == 3:
        resized_all = resize_fn(data, rec_list[0].shape[2] ,type = resize_method)
        resized_small = resize_fn(data, rec_list[1].shape[2], type=resize_method)
        resized_part =resize_fn(crop_image_by_part(data, part),  rec_list[2].shape[2], type = resize_method)
        if aux_loss_fn == 'percept':
            aux_loss = percept(rec_list[0], resized_all).sum() +\
                        percept(rec_list[1], resized_small).sum() +\
                        percept(rec_list[2], resized_part).sum()
        elif aux_loss_fn == 'mse':
            aux_loss = F.mse_loss(rec_list[0], resized_all) + \
                        F.mse_loss(rec_list[1], resized_small) + \
                        F.mse_loss(rec_list[2], resized_part)
        elif aux_loss_fn == 'combi':
            aux_loss = percept(rec_list[0], resized_all).sum() +\
                        percept(rec_list[1], resized_small).sum() +\
                        percept(rec_list[2], resized_part).sum()
            aux_loss += mse_scaling*(F.mse_loss(rec_list[0], resized_all) + \
                        F.mse_loss(rec_list[1], resized_small) + \
                        F.mse_loss(rec_list[2], resized_part))

    else:
        resized_all = resize_fn(data, rec_list[0].shape[2] ,type = resize_method)
        resized_part =resize_fn(crop_image_by_part(data, part),  rec_list[1].shape[2], type = resize_method)
        if aux_loss_fn == 'percept':
            aux_loss = percept(rec_list[0], resized_all).sum() +\
                        percept(rec_list[1], resized_part).sum()
        elif aux_loss_fn == 'mse':
            aux_loss = F.mse_loss(rec_list[0], resized_all) + \
                        F.mse_loss(rec_list[1], resized_part)
        elif aux_loss_fn == 'combi':
            aux_loss = percept(rec_list[0], resized_all).sum() +\
                        percept(rec_list[1], resized_part).sum()
            aux_loss += mse_scaling*(F.mse_loss(rec_list[0], resized_all) + \
                        F.mse_loss(rec_list[1], resized_part))

    return aux_loss

def train_d(net, iter, train_config, ema_losses, data_real, data_fake,
            amp_context, optimizerD, scalerD, d_loss_fn, aux_dict):

    with amp_context():
        D_real, rec_list, part = net(data_real, "real")
        D_fake = net(data_fake, "fake")

    D_loss_real, D_loss_fake = d_loss_fn(D_fake, D_real, ema_losses, iter, train_config['label_smooth'])

    # auxiliary loss
    D_loss_aux = calc_aux_loss(data_real, part, rec_list, aux_dict['aux_loss_fn'], aux_dict['resize_method'], aux_dict['mse_scaling'])

    # LeCam loss
    if train_config['LC'] > 0 and iter > ema_losses.start_iter:
        D_loss_LC = losses.lecam_reg(D_real, D_fake, ema_losses)*train_config['LC']
    else:
        D_loss_LC = torch.tensor(0.)

    D_loss_total = D_loss_real + D_loss_fake + D_loss_aux + D_loss_LC

    scalerD.scale(D_loss_total).backward()
    scalerD.step(optimizerD)
    scalerD.update()

    train_metrics = {'D_loss_total': D_loss_total.item(),
                    'D_loss_real': D_loss_real.item(),
                    'D_loss_fake': D_loss_fake.item(),
                    'D_loss_aux': D_loss_aux.item(),
                    'D_real': torch.mean(D_real).item(),
                    'D_fake': torch.mean(D_fake).item()
    }
    if train_config['LC'] > 0:
        train_metrics['D_loss_LC'] = D_loss_LC.item()

    return train_metrics, rec_list[0], rec_list[-1]

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    with_fid = args.fid
    train_config = args.train_config
    model_config = args.model_config
    with_amp = args.with_amp
    log_every = args.log_every
    # load training config
    train_config = get_config('training_configs.csv', train_config, type='train')
    print(train_config)

    # Use TTUR if needed
    g_lr = train_config['g_lr']
    d_lr = train_config['d_lr']

    nbeta1 = train_config['nbeta1']
    policy = train_config['policy']
    aug_prob = train_config['aug_prob']
    aux_loss = train_config['aux_loss']
    resize_method = train_config['resize_method']
    mse_scaling = train_config['mse_scaling']
    aux_dict = {'aux_loss_fn': aux_loss,
                'resize_method': resize_method,
                'mse_scaling': mse_scaling
    }

    # set up adversarial loss
    adv_loss = train_config['adv_loss']
    if adv_loss == 'rahinge':
        d_loss_fn = losses.loss_rahinge_dis
        g_loss_fn = losses.loss_rahinge_gen
    elif adv_loss == 'hinge':
        d_loss_fn = losses.loss_hinge_dis
        g_loss_fn = losses.loss_hinge_gen

    # set up EMA object to track loss and discriminator predictions
    loss_ema = train_config['loss_ema']
    if loss_ema:
        ema_losses = losses.ema_losses(start_iter=train_config['loss_ema_start'])
        LC = train_config['LC']
    else:
        ema_losses = None
        LC = 0.

    use_cuda = True
    dataloader_workers = 2
    current_iteration = 0
    save_interval = 1000

    # load model configs
    model_config = get_config('model_configs.csv', model_config, type='model')
    nz = model_config['nz']
    d_num_rec = model_config['d_num_rec']
    # print(model_config) #print model configuration to the terminal

    saved_model_folder, saved_image_folder, fid_folder = get_dir(args)
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [NumpyRandomFlip(),
                      NumpyToTensor(),
                      transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                      ]

    transform_pil_list = [transforms.ToPILImage(),
                             transforms.Resize(128),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                             ]

    trans = transforms.Compose(transform_list)
    trans_pil = transforms.Compose(transform_pil_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    elif 'hdf5' in data_root:
        dataset = H5Dataset(file_path=data_root, transform=trans)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))

    # load in vgg-lin for perceptual distance calculation in auxiliary task loss
    global percept
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

    # Initialise Generator
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
    netG.apply(weights_init)

    # Initialise Discriminator
    netD = Discriminator(
                         activation         = model_config['d_activation'],
                         chan_attn          = model_config['d_chan_attn'],
                         sle_map            = model_config['d_skip_map'],
                         skip_conn          = model_config['d_skip_conn'],  # skip-layer connections to use. Original paper has none in discriminator
                         spatial_attn       = model_config['d_spatial_attn'],
                         attn_layers        = model_config['d_attn_layers'],
                         attn_dis2          = model_config['d_2_attn'],
                         anti_alias         = model_config['d_anti_alias'],
                         out_logits_size    = model_config['d_out_logits_size'],        # alternative is 1
                         multi_logits_out   = model_config['d_multi_logits_out'],
                         small_im_size      = model_config['d_small_im_size'],      # should be same as generator
                         d_num_rec          = model_config['d_num_rec'],
                         use_tanh           = model_config['use_tanh'],
                         minibatch_d        = model_config['minibatch_d'],


    )
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    # initialise copy for storing EMA weights of generator
    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(nbeta1, 0.999))
    scalerG = amp.GradScaler(enabled = with_amp)
    scalerD = amp.GradScaler(enabled = with_amp)

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        fixed_noise = ckpt['fixed_noise']
        ema_losses = ckpt['ema_losses']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        scalerG.load_state_dict(ckpt['scaler_g'])
        scalerD.load_state_dict(ckpt['scaler_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])+1
        netG.train()
        netD.train()
        del ckpt

    # initialise logger for storing losses and discriminator outputs during training
    train_log = GANLogger(file_dir = saved_model_folder,
                            reinitialize = (checkpoint is None),
                            filename='train_log.csv')

    amp_context = amp.autocast if with_amp else null_context
    for iteration in tqdm(range(current_iteration, total_iterations+1)):

        real_image = next(dataloader)
        real_image = real_image.cuda(non_blocking=True)

        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        with amp_context():
            fake_images = netG(noise)
            # determine whether to apply differential augmentation
            if torch.rand(1)<aug_prob:
                real_image = DiffAugment(real_image, policy=policy)
                fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        ## 2. train Discriminator
        netD.zero_grad()
        train_metrics, rec_img_all, rec_img_part = train_d(netD, iteration,
                                                    train_config, ema_losses, real_image,
                                                    [fi.detach() for fi in fake_images],
                                                    amp_context, optimizerD, scalerD,
                                                    d_loss_fn, aux_dict)

        ## 3. train generator
        netG.zero_grad()
        with amp_context():
            D_fake = netD(fake_images, "fake")
            D_real = train_metrics['D_real']

        G_loss = g_loss_fn(D_fake, torch.Tensor([D_real]).to(device), train_config['label_smooth'])
        scalerG.scale(G_loss).backward()
        # update EMA of G_loss.
        if loss_ema:
            ema_losses.update(G_loss.item(), 'G_loss', iteration)
        scalerG.step(optimizerG)
        scalerG.update()

        # log the losses etc. for plotting
        if iteration % log_every == 0:
            train_metrics['iter'] = iteration
            train_metrics['G_loss'] = G_loss.item()
            train_log.log(train_metrics)

        # update EMA version of Generator
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            D_loss_out = train_metrics['D_loss_total']
            print(f"GAN: loss_d_total: {D_loss_out:.5f}   loss_g: {G_loss.item():.5f}")

        # save images produced using the fixed noise, as well as cropped and full reconstruction images
        if iteration % (save_interval) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128),
                        rec_img_all,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)

        # save the current training state
        if iteration % (save_interval*5) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict(),
                        'scaler_g': scalerG.state_dict(),
                        'scaler_d': scalerD.state_dict(),
                        'ema_losses': ema_losses,
                        'fixed_noise': fixed_noise}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='./train_data/cored_train.h5', help='path of resource dataset (hdf5 if possible)')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--train_config', type=str, default='base', help='select training configuration (see csv). base = original FastGAN')
    parser.add_argument('--model_config', type=str, default='base', help='select model configuration (see csv). base = original FastGAN')
    parser.add_argument('--with_amp', type=int, default=0, help='use AMP training? Only do so if model size is an issue')
    parser.add_argument('--log_every', type=int, default=10, help='how often to record losses etc.')
    parser.add_argument('--calc_metrics', type=int, default=0, help='whether to evaluate prdc, 1NN and KID. Not currently used.')
    parser.add_argument('--eval_every', type=int, default=1000, help='how often to calculate evaluation metrics. Not currently used.')

    args = parser.parse_args()
    print(args)

    # do the training
    train(args)

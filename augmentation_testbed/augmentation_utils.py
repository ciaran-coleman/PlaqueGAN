import torch
import numpy as np

# https://github.com/facebookresearch/mixup-cifar10
# https://github.com/hellowangqian/multi-label-image-classification
# def mixup_data(x, y, alpha=1.0, use_cuda=True):
#     """
#     Returns mixed inputs, pairs of targets, and lambda
#     """
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#
#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)
#
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

# mix random pairs from all classes
# https://github.com/facebookresearch/mixup-cifar10
# https://github.com/hellowangqian/multi-label-image-classification
# include option to have non-binary label
def mixup_data(x,y, alpha = 0, label_construct = 'ave', device = torch.device('cpu')):
    """
    Apply different versions of mixup.
    alpha in [0.1, 0.4], label_construct = 'ave' --> original mixup applied to multi-label
    alpha = 0, label_construct = 'union' --> mixup adapted to multi-label in hellowangqian
    alpha = 0, label_construct = 'orig' --> SamplePairing strategy
    """
    batch_size = x.shape[0]

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5

    if device == torch.device('cuda'):
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam*x + lam*x[index,:]

    if label_construct == 'ave':
        mixed_y = lam*y + lam*y[index,:]
    elif label_construct == 'union':
        mixed_y = (y + y[index,:])>0
        mixed_y = mixed_y.float()
    elif label_construct == 'orig': # keep original label
        mixed_y = y
    return mixed_x, mixed_y

def construct_mixup_args(aug_type):
    if aug_type == 'standard' or aug_type=='none':
        mixup_args = {'mix': False}

    elif aug_type == 'mixup':
        mixup_args = {'mix': True,
                      'alpha': 0.2,
                      'label_construct': 'ave'}

    elif aug_type == 'mixup_alt':
        mixup_args = {'mix': True,
                      'alpha': 0,
                      'label_construct': 'union'}

    elif aug_type == 'samplepairing':
        mixup_args = {'mix': True,
                      'alpha': 0,
                      'label_construct': 'orig'}

    elif aug_type == 'gan':
        mixup_args = {'mix': False}

    return mixup_args

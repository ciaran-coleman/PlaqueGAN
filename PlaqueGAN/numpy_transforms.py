import torch
import numpy as np
import random
from torchvision import transforms

class NumpyRandomFlip(object):

    def __init__(self, p = 0.5):
        self.p = p
        random.seed(None)

    def __call__(self, img):

        if random.random() < self.p:
            return np.flip(img,axis=0).copy()
        if random.random() < self.p:
            return np.flip(img,axis=1).copy()
        return img

class NumpyToTensor(object):

    def __init__(self):
        return

    def __call__(self, img):
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1],1)

        return transforms.functional.to_tensor(img)

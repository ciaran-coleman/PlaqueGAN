import torch
import torch.nn as nn
from torchvision import models

class VGG16Features(nn.Module):
    """
    Build desired VGG16 model for the purpose of extracting features
    """

    def __init__(self, device=torch.device("cpu"), pre_train=True, fc_dim = 64):
        super(VGG16Features, self).__init__()
        # Load and partition the model
        if pre_train:
            vgg16 = models.vgg16(pretrained=True).to(device)
        else:
            vgg16 = models.vgg16(pretrained=False).to(device)

        self.vgg16_features = vgg16.features
        self.avgpool = vgg16.avgpool

        if fc_dim == 4096:
            self.classifier = vgg16.classifier[:-2]
        else:
            classifier_begin = vgg16.classifier[:-4]
            classifier_end = nn.Sequential(nn.Linear(in_features=4096, out_features = fc_dim),
                                          nn.ReLU(inplace=True))

            self.classifier = nn.Sequential(*(list(classifier_begin)+list(classifier_end)))

    def forward(self, x):
        x = self.vgg16_features(x)
        x = self.avgpool(x)
        x = x.view(-1, 7 * 7 * 512)
        x = self.classifier(x)
        return x

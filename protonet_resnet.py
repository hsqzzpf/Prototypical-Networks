import torch.nn as nn
from torchvision import models

class ProtoResNet(nn.Module):
    '''
    Resnet50 model with fine tuning

    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoResNet, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        model_ft.conv1=nn.Conv2d(x_dim, hid_dim, kernel_size=(3, 3), bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, z_dim)
        self.encoder = model_ft

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

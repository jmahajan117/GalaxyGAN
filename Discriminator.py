import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        input_channels = 3
        slope = 0.2

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 128, 4, 2, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(slope),
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(slope),
            nn.Conv2d(512, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(slope),
            nn.Conv2d(1024, 1, 4, 1, padding=0),
        )

    def forward(self, x):
        return self.model(x)
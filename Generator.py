import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    
    def __init__(self, noise_dim):
        super().__init__()

        self.noise_dim = noise_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, 4, 1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 3, 4, 2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
      x = x.view(-1, self.noise_dim, 1, 1)
      return self.model(x)

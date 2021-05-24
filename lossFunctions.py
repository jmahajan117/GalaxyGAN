import torch
import torch.nn as nn
import numpy as np

def discrim_loss(real, fake, device):
    loss = 0.5 * torch.mean(torch.pow(real - 1, 2)) + \
           0.5 * torch.mean(torch.pow(fake, 2))
    return loss

def gen_loss(fake, device):
    loss = 0.5 * torch.mean(torch.pow(fake - 1, 2))
    return loss
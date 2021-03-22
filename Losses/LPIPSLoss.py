import kornia
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .lpips_pytorch import LPIPS

class LPIPSLoss(nn.Module):
    def __init__(self,):
        super(LPIPSLoss, self).__init__()
        self.criterion = LPIPS(
                net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
                version='0.1'    # Currently, v0.1 is supported
            )

    def forward(self, x, y, FValid):
        loss = self.criterion(x, y, FValid)

        return loss

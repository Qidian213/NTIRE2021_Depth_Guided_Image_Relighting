import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .ColorFunction import *

class GrayLoss(nn.Module):
    def __init__(self,):
        super(GrayLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):  ### BGR->Gray
        x = bgr_to_grayscale(x)
        y = bgr_to_grayscale(y)
        
        loss = self.criterion(x, y)

        return loss
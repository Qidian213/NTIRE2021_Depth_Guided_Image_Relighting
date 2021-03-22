import torch 
import torch.nn as nn 
import torch.nn.functional as F

class SmoothL1Loss(nn.Module):
    def forward(self, x, t, sigma=1):
        sigma2 = sigma ** 2

        diff = (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
        
        return y.mean()
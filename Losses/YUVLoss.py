import kornia
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class YUVLoss(nn.Module):
    def __init__(self,):
        super(YUVLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):  ### RGB->YUV
        x = kornia.bgr_to_yuv(x)
        y = kornia.bgr_to_yuv(y)
        
        loss = self.criterion(x[:,1:,:,:], y[:,1:,:,:])

        return loss
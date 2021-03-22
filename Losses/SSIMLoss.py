import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class SSIMLoss(nn.Module):
    def __init__(self, ):
        super(SSIMLoss, self).__init__()
        self.ce_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=False, channel=3)

    def forward(self, inputs, targets, FVaild):
        loss = self.ce_loss(inputs, targets)*FVaild
        return loss.mean()
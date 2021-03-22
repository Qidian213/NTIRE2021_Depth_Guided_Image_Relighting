import torch
import torch.nn as nn
from torch.nn.functional import mse_loss as mse

def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    return 10. * torch.log10(max_val ** 2 / mse(input, target, reduction='mean'))
    
class PSNR(nn.Module):
    def __init__(self, max_val: float) -> None:
        super(PSNR, self).__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr(input, target, self.max_val)

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_gauss(img, kernel):
    # conv img with a gaussian kernel that has been built with build_gauss_kernel
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)

def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff     = current - filtered
        pyr.append(diff)
        current  = F.avg_pool2d(filtered, 2)
    pyr.append(current)
    
    return pyr
    
class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size     = k_size
        self.sigma      = sigma
        self._gauss_kernel = None
        self.L1_loss    = nn.L1Loss()

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(size=self.k_size, sigma=self.sigma, n_channels=input.shape[1], cuda=input.is_cuda)

        pyr_input  = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(self.L1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


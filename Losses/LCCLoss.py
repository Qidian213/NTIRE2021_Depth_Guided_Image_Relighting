import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .ColorFunction import *

class LCCLoss(nn.Module):
    """ 
    local (over window) normalized cross correlation (square)
    """ 
    def __init__(self, win=[9, 9], eps=1e-5):
        super(LCCLoss, self).__init__()
        self.win = win
        self.eps = eps
        
    def forward(self, I, J):
        I = bgr_to_grayscale(I)
        J = bgr_to_grayscale(J)
        
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        
        filters = Variable(torch.ones(1, 1, self.win[0], self.win[1]))
        filters = filters.cuda()
        padding = (self.win[0]//2, self.win[1]//2)
        
        I_sum  = F.conv2d(I, filters, stride=1, padding=padding)
        J_sum  = F.conv2d(J, filters, stride=1, padding=padding)
        I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
        J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
        IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)
        
        win_size = self.win[0]*self.win[1]
 
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
 
        cc = cross*cross / (I_var*J_var + self.eps)
        lcc = -1.0 * torch.mean(cc) + 1
        
        return lcc
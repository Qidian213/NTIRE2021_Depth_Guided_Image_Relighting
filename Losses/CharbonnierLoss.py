import torch
import torch.nn as nn
import torch.nn.functional as F

class L1_Charbonnier_loss(nn.Module):
    def __init__(self, ):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff  = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss  = torch.mean(error)
        return loss

class L1_Charbonnier_loss_color(nn.Module):
    def __init__(self, ):
        super(L1_Charbonnier_loss_color, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss
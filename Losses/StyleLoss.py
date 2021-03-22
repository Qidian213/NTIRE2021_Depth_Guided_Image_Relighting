import torch 
import torch.nn as nn 
import torch.nn.functional as F

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, x, y):
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        y = y.view(y.size(0), y.size(1), y.size(2) * y.size(3))
        x_style = torch.matmul(x, x.transpose(2, 1))
        y_style = torch.matmul(y, y.transpose(2, 1))
        loss_value = torch.mean(torch.abs(x_style - y_style))
        return loss_value
import torch 
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, num_class=5,  gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps   = eps
        self.num_class = num_class

    def forward(self, input, target, CVaild):
        one_hot = F.one_hot(target, self.num_class).float()
        
        logp    = F.softmax(input, dim=1)
        
        focal_p = - (1-logp)**self.gamma * torch.log(logp)
        loss_p  = torch.sum(one_hot * focal_p, dim=1)*CVaild
        return loss_p.mean()
        
    # def forward(self, input, target, CVaild):
        # one_hot = F.one_hot(target, self.num_class).float()
        
        # logp    = F.softmax(input, dim=1)
        
        # focal_p = - (1-logp)**self.gamma * torch.log(logp)
       # # focal_n = - logp**self.gamma * torch.log(1-logp)
        
        # loss_p = torch.sum(one_hot * focal_p, dim=1)
       # # loss_n = torch.sum((1-one_hot) * focal_n, dim=1)
       # # return (loss_p+loss_n).mean()
        # return loss_p.mean()
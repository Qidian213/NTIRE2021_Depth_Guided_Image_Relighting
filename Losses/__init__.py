import kornia
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .GradLoss import L1GradientLoss
from .GrayLoss import GrayLoss
from .LCCLoss import LCCLoss
from .SmoothL1Loss import SmoothL1Loss
from .YUVLoss import YUVLoss
from .LPIPSLoss import LPIPSLoss
from .LapLoss import LapLoss
from .GANLoss import *
from .StyleLoss import StyleLoss
from .LightLoss import LightLoss
from .PSNRLoss import PSNR
from .FocalLoss import FocalLoss
from .SSIMLoss import SSIMLoss

class CrossEntropyLoss(nn.Module):
    def __init__(self, ):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, labels, CVaild):
        loss = self.ce_loss(inputs, labels)*CVaild
        return loss.mean()

class MSELoss(nn.Module):
    def __init__(self, ):
        super(MSELoss, self).__init__()
        self.ce_loss = nn.MSELoss(reduction='none')
        
    def forward(self, inputs, targets, FVaild):
        loss = self.ce_loss(inputs, targets)
        loss = loss.mean(dim=(1,2,3))*FVaild      
        return loss.mean()

class L1Loss(nn.Module):
    def __init__(self, ):
        super(L1Loss, self).__init__()
        self.ce_loss = nn.L1Loss(reduction='none')
        
    def forward(self, inputs, targets, FVaild):
        loss = self.ce_loss(inputs, targets)
        loss = loss.mean(dim=(1,2,3))*FVaild 
        return loss.mean()
        
def Get_LossFunction(cfgs):
    """ return given loss function
    """
    loss_fun ={}
    loss_fun['criterionCE']    = CrossEntropyLoss()
    loss_fun['criterionFDIR']  = FocalLoss(num_class=8)
    loss_fun['criterionFTMP']  = FocalLoss(num_class=5)
    loss_fun['criterionL1']    = L1Loss()
    loss_fun['criterionMSE']   = MSELoss()
    loss_fun['criterionSML1']  = SmoothL1Loss()
    loss_fun['criterionLCC']   = LCCLoss()
    loss_fun['criterionGray']  = GrayLoss()
    loss_fun['criterionLight'] = LightLoss()
    loss_fun['criterionGrad']  = L1GradientLoss()
    loss_fun['criterionStyle'] = StyleLoss()
    loss_fun['criterionSSIM']  = SSIMLoss()
    loss_fun['criterionYUV']   = YUVLoss()
    loss_fun['criterionPSNR']  = PSNR(1.0)
    
    if(cfgs.useLPIPS): 
        loss_fun['criterionLPIPS'] = torch.nn.DataParallel(LPIPSLoss().cuda())
    return loss_fun


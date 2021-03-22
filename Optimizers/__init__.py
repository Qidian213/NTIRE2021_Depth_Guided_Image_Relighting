import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .schedulers import WarmUpLR, WarmupMultiStepLR, WarmupMultiEpochLR
from .lr_scheduler import LR_Scheduler

def Get_Optimizer(cfgs, model, iter_per_epoch):
    optim_scheduler = {}
    
    ### optimizer
    if(cfgs.Optim_Type == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=cfgs.Lr_Base, weight_decay=1e-4)
        
    if(cfgs.Optim_Type == 'Momentum'):
        optimizer = optim.SGD(model.parameters(), lr=cfgs.Lr_Base, momentum=0.9, weight_decay=1e-4)
        
    if(cfgs.Optim_Type == 'Adam'):
        optimizer = optim.Adam(model.parameters(), lr=cfgs.Lr_Base, betas=(0.9, 0.99), weight_decay=1e-4)

    if(cfgs.Optim_Type == 'AdamW'):
        optimizer = optim.AdamW(model.parameters(), lr=cfgs.Lr_Base, betas=(0.9, 0.99), weight_decay=1e-4)
        
    ### lr_scheduler
    if(cfgs.Sche_Type == "WarmupMultiStepLR"):
        scheduler = WarmupMultiStepLR(optimizer, iter_per_epoch, cfgs.Warmup_epoch, cfgs.Lr_Adjust)

    if(cfgs.Sche_Type == "WarmupMultiEpochLR"):
        scheduler = WarmupMultiEpochLR(optimizer, iter_per_epoch, cfgs.Warmup_epoch, cfgs.Lr_Adjust)
        
    if(cfgs.Sche_Type == "LambdaLR"):
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

    optim_scheduler['optimizer'] = optimizer
    optim_scheduler['scheduler'] = scheduler

    return optim_scheduler

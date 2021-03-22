from bisect import bisect_right
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class WarmupMultiEpochLR(_LRScheduler):
    def __init__(self, optimizer, 
                    epoch_iters, 
                    warmstones, 
                    milestones, 
                    warmup_factor=0.01, 
                    gamma=0.1, 
                    warmup_method="constant",
                    last_epoch=-1
                ):
        
        self.milestones    = [item for item in milestones] # Counter(milestones)
        self.gamma         = gamma
        self.warmup_factor = warmup_factor
        self.warmstones    = warmstones 
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if(self.last_epoch < self.warmstones):
            if(self.warmup_method == "linear"):
                alpha = self.last_epoch / self.warmstones
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                warmup_factor = self.warmup_factor
                
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, 
                    epoch_iters, 
                    warmstones, 
                    milestones, 
                    warmup_factor=-0.01, 
                    gamma=0.1, 
                    last_epoch=-1
                ):
        
        self.milestones    = [item*epoch_iters for item in milestones] # Counter(milestones)
        self.gamma         = gamma
        self.warmup_factor = warmup_factor
        self.epoch_iters   = epoch_iters
        self.warmup_iters  = epoch_iters*warmstones + 1e-8
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if(self.last_epoch < self.warmup_iters):
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

from torch.utils.data import DataLoader
from .LdDataset import CreateDataset

def CreateDataLoader(opts):
    train_dataset = CreateDataset(opts, mode = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batchSize, shuffle= True, num_workers=int(opts.nThreads))
    
    val1_dataset  = CreateDataset(opts, mode = 'val1')
    val1_dataloader = DataLoader(val1_dataset, batch_size=1, shuffle= False, num_workers=0)

    val2_dataset  = CreateDataset(opts, mode = 'val2')
    val2_dataloader = DataLoader(val2_dataset, batch_size=1, shuffle= False, num_workers=0)

    test1_dataset  = CreateDataset(opts, mode = 'test1')
    test1_dataloader = DataLoader(test1_dataset, batch_size=1, shuffle= False, num_workers=0)

    test2_dataset  = CreateDataset(opts, mode = 'test2')
    test2_dataloader = DataLoader(test2_dataset, batch_size=1, shuffle= False, num_workers=0)
    
    return train_dataloader, val1_dataloader, val2_dataloader, test1_dataloader, test2_dataloader

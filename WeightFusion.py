import os
import torch
import numpy as np
from collections import OrderedDict
from Models import CreateModel

class BaseOptions():
    def __init__(self):
        # experiment specifics
        self.model = 'DFuseNet' 
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    opts = BaseOptions()
    
    Remodel = CreateModel(opts)

    model_file1 = '/data/zzg/NTIRE_Relight/NTIRE_Relighting/work_space/DFuseNet_2021-03-17-11-33-34/Model/Epoch_158.pth'
    checkpoint1 = torch.load(model_file1, map_location=torch.device('cpu'))

    model_file2 = '/data/zzg/NTIRE_Relight/NTIRE_Relighting/work_space/DFuseNet_2021-03-17-11-33-34/Model/Epoch_108.pth'
    checkpoint2 = torch.load(model_file2, map_location=torch.device('cpu'))

    model_file3 = '/data/zzg/NTIRE_Relight/NTIRE_Relighting/work_space/DFuseNet_2021-03-17-11-33-34/Model/Epoch_186.pth'
    checkpoint3 = torch.load(model_file2, map_location=torch.device('cpu'))
    
    scale = 1/3.0
    state_dict = OrderedDict()
    for key, tensor in checkpoint1.items():
        key = key.replace('module.', '')
        state_dict[key] = tensor*scale

    for key, tensor in checkpoint2.items():
        key = key.replace('module.', '')
        state_dict[key] += tensor*scale

    for key, tensor in checkpoint3.items():
        key = key.replace('module.', '')
        state_dict[key] += tensor*scale
        
    Remodel.load_state_dict(state_dict)
    torch.save(Remodel.state_dict(),'DFuseNet.pth')


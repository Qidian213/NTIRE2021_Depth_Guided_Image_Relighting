import torch
from .DFuseNet import DFuseNet
from .BFuseNet import BFuseNet
from .RDFuseNet import RDFuseNet

def CreateModel(opts):
    if(opts.model == 'DFuseNet'):
        model = DFuseNet()

    if(opts.model == 'BFuseNet'):
        model = BFuseNet()

    if(opts.model == 'RDFuseNet'):
        model = RDFuseNet()
        
    return model
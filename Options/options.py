import os

class BaseOptions():
    def __init__(self):
        # experiment specifics
        self.Save_dir        = 'work_space'
        self.model           = 'DFuseNet'   #RDFuseNet, DFuseNet
        self.preTrained      = ''

        # input/output sizes       
        self.inputSize       = 512
        self.guideSize       = 512
        self.targetSize      = 512
        
        # dataloader
        self.batchSize      = 16
        self.nThreads       = 4

        # optim
        self.Optim_Type     = 'Adam'
        self.Lr_Base        = 1e-4
        self.Sche_Type      = 'WarmupMultiEpochLR'
        self.Lr_Adjust      = [80, 160, 240]
        self.Warmup_epoch   = 1
        self.max_epoch      = 280
        
        # Loss
        self.useLPIPS       = False
        self.useGrad        = True
        
        # dataset
        self.train_input_folder  = "/data/hejy/datasets/NRITE_Relight/Track2/train/"      ## path to train dataset of track2
        self.train_guide_folder  = "/data/hejy/datasets/NRITE_Relight/Track2/train/"  
        self.train_target_folder = "/data/hejy/datasets/NRITE_Relight/Track2/train/"  

        self.val1_input_folder   = "/data/hejy/datasets/NRITE_Relight/Track1/val/"        ## path to val input dataset of track1
        self.val1_guide_folder   = "/data/hejy/datasets/NRITE_Relight/Track2/train/"      ## path to train dataset of track2
        self.val1_target_folder  = "/data/hejy/datasets/NRITE_Relight/Track1/val_target/" ## path to val target dataset of track1

        self.val2_input_folder   = "/data/hejy/datasets/NRITE_Relight/Track2/val/input/"  ## path to val input dataset of track2
        self.val2_guide_folder   = "/data/hejy/datasets/NRITE_Relight/Track2/val/guide/"  ## path to val guide dataset of track2
        self.val2_target_folder  = "/data/hejy/datasets/NRITE_Relight/Track2/val/target/" ## path to val target dataset of track2

        self.test1_input_folder  = "/data/hejy/datasets/NRITE_Relight/Track1/test/"       ## path to test input dataset of track1
        self.test1_guide_folder  = "/data/hejy/datasets/NRITE_Relight/Track2/train/"      ## path to train dataset of track2

        self.test2_input_folder  = "/data/hejy/datasets/NRITE_Relight/Track2/test/input/" ## path to test input dataset of track2
        self.test2_guide_folder  = "/data/hejy/datasets/NRITE_Relight/Track2/test/guide/" ## path to test guide dataset of track2
        

import os 
import cv2
import random 
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset

def randrf(low, high):
    return random.uniform(0, 1) * (high - low) + low
    
class CreateDataset(Dataset):
    def __init__(self, opts, mode):
        super(CreateDataset, self).__init__()
        self.opts = opts
        self.mode = mode
        self.inputSize  = opts.inputSize
        self.guideSize  = opts.guideSize
        self.targetSize = opts.targetSize
        
        self.train_input_folder  = opts.train_input_folder
        self.train_guide_folder  = opts.train_guide_folder
        self.train_target_folder = opts.train_target_folder
        self.train_filenames     = [['train', x] for x in os.listdir(self.train_input_folder) if self.is_image_file(x)]

        self.val1_input_folder   = opts.val1_input_folder
        self.val1_guide_folder   = opts.val1_guide_folder
        self.val1_target_folder  = opts.val1_target_folder
        self.val1_filenames      = [['val1', x] for x in os.listdir(self.val1_input_folder) if self.is_image_file(x)]

        self.val2_input_folder   = opts.val2_input_folder
        self.val2_guide_folder   = opts.val2_guide_folder
        self.val2_target_folder  = opts.val2_target_folder
        self.val2_filenames      = [['val2', x] for x in os.listdir(self.val2_input_folder) if self.is_image_file(x)]

        self.test1_input_folder  = opts.test1_input_folder
        self.test1_guide_folder  = opts.test1_guide_folder
        self.test1_filenames     = [['test1', x] for x in os.listdir(self.test1_input_folder) if self.is_image_file(x)]
 
        self.test2_input_folder  = opts.test2_input_folder
        self.test2_guide_folder  = opts.test2_guide_folder
        self.test2_filenames     = [['test2', x] for x in os.listdir(self.test2_input_folder) if self.is_image_file(x)]

        if(self.mode == 'train'):
            self.image_filenames = self.train_filenames
        if(self.mode == 'val1'):
            self.image_filenames = self.val1_filenames
        if(self.mode == 'val2'):
            self.image_filenames = self.val2_filenames 
        if(self.mode == 'test1'):
            self.image_filenames = self.test1_filenames
        if(self.mode == 'test2'):
            self.image_filenames = self.test2_filenames 
            
        self.tmp2id  = {'2500':0, '3500':1, '4500':2, '5500':3, '6500':4}
        self.dir2id  = {'E':0, 'N':1, 'NE':2, 'NW': 3, 'S':4, 'SE':5, 'SW':6, 'W':7}
        self.HorFlip = {'E':'W', 'N':'N', 'NE':'NW', 'NW':'NE', 'S':'S', 'SE':'SW', 'SW':'SE', 'W':'E'}
        self.VerFlip = {'E':'E', 'N':'S', 'NE':'SE', 'NW':'SW', 'S':'N', 'SE':'NE', 'SW':'NW', 'W':'W'}

        print(f"Data num: {len(self.image_filenames)}")
        
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

    def __getitem__(self, index):
        if(self.mode == 'train'):
            mode, InFileName = self.image_filenames[index]
            InFile      = self.train_input_folder + InFileName
            InDepthFile = self.train_input_folder + InFileName[:8] + '.npy'
            mode, GdFileName = random.choice(self.train_filenames )
            GdFile      = self.train_guide_folder + GdFileName
            GdDepthFile = self.train_guide_folder + GdFileName[:8] + '.npy'
            GtFile      = self.train_target_folder + InFileName[:8] + GdFileName[8:]
        
            InFileNameS = InFileName.split('_')
            InTmp = InFileNameS[1]
            InDir = InFileNameS[2].replace('.png', '')

            GdFileNameS = GdFileName.split('_')
            GdTmp = GdFileNameS[1]
            GdDir = GdFileNameS[2].replace('.png', '')

            GtFileNameS = GdFileName.split('_')
            GtTmp = GtFileNameS[1]
            GtDir = GtFileNameS[2].replace('.png', '')
            
            InTmp = self.tmp2id[InTmp]
            InDir = self.dir2id[InDir]
            GdTmp = self.tmp2id[GdTmp]
            GdDir = self.dir2id[GdDir]
            GtTmp = self.tmp2id[GtTmp]
            GtDir = self.dir2id[GtDir]
            CVaild = 1
            FValid = 1
                
            Img_Input  = cv2.imread(InFile)
            Img_Input  = cv2.resize(Img_Input, (self.inputSize, self.inputSize))    ###
            Img_Guide  = cv2.imread(GdFile)
            Img_Guide  = cv2.resize(Img_Guide, (self.guideSize, self.guideSize))    ###
            Img_Target = cv2.imread(GtFile)
            Img_Target = cv2.resize(Img_Target, (self.targetSize, self.targetSize))   ###
            depth_information = np.load(InDepthFile, allow_pickle=True)
            Depth_Input       = depth_information.item().get('normalized_depth')
            Depth_Input       = cv2.resize(Depth_Input, (self.inputSize, self.inputSize)) ###
            depth_information = np.load(GdDepthFile, allow_pickle=True)
            Depth_Guide       = depth_information.item().get('normalized_depth')
            Depth_Guide       = cv2.resize(Depth_Guide, (self.guideSize, self.guideSize)) ###

            Img_Input  = Img_Input.transpose((2,0,1))
            Img_Input  = torch.from_numpy(Img_Input.astype(np.float32)/255.0)

            Img_Guide  = Img_Guide.transpose((2,0,1))
            Img_Guide  = torch.from_numpy(Img_Guide.astype(np.float32)/255.0)

            Img_Target = Img_Target.transpose((2,0,1))
            Img_Target = torch.from_numpy(Img_Target.astype(np.float32)/255.0)
            
            Depth_Input = T.ToTensor()(Depth_Input)
            Depth_Guide = T.ToTensor()(Depth_Guide)

        if(self.mode == 'val1'):
            mode, InFileName  = self.image_filenames[index]
            InFile      = self.val1_input_folder + InFileName
            InDepthFile = self.val1_input_folder + InFileName[:8] + '.npy'
            GdFile      = self.val1_guide_folder + 'Image049_4500_E.png'
            GdDepthFile = self.val1_guide_folder + 'Image049.npy'
            GtFile      = self.val1_target_folder + InFileName

            Img_Input  = cv2.imread(InFile)
            Img_Input  = cv2.resize(Img_Input, (self.inputSize, self.inputSize))
            Img_Input  = Img_Input.transpose((2,0,1))
            Img_Input  = torch.from_numpy(Img_Input.astype(np.float32)/255.0)
            
            Img_Guide  = cv2.imread(GdFile)
            Img_Guide  = cv2.resize(Img_Guide, (self.guideSize, self.guideSize))
            Img_Guide  = Img_Guide.transpose((2,0,1))
            Img_Guide  = torch.from_numpy(Img_Guide.astype(np.float32)/255.0)

            Img_Target = cv2.imread(GtFile)
            Img_Target = cv2.resize(Img_Target, (self.targetSize, self.targetSize))
            Img_Target = Img_Target.transpose((2,0,1))
            Img_Target = torch.from_numpy(Img_Target.astype(np.float32)/255.0)
            
            depth_information = np.load(InDepthFile, allow_pickle=True)
            Depth_Input = depth_information.item().get('normalized_depth')
            Depth_Input = cv2.resize(Depth_Input, (self.inputSize, self.inputSize))
            Depth_Input = T.ToTensor()(Depth_Input)
            
            depth_information = np.load(GdDepthFile, allow_pickle=True)
            Depth_Guide = depth_information.item().get('normalized_depth')
            Depth_Guide = cv2.resize(Depth_Guide, (self.guideSize, self.guideSize))
            Depth_Guide = T.ToTensor()(Depth_Guide)

            InTmp  = 4
            InDir  = 1
            GdTmp  = 2
            GdDir  = 0
            GtTmp  = -1
            GtDir  = -1
            CVaild = -1
            FValid = -1
            
        if(self.mode == 'val2'):
            mode, InFileName  = self.image_filenames[index]
            InFile      = self.val2_input_folder + InFileName
            InDepthFile = self.val2_input_folder + InFileName[:7] + '.npy'
            GdFile      = self.val2_guide_folder + InFileName
            GdDepthFile = self.val2_guide_folder + InFileName[:7] + '.npy'
            GtFile      = self.val2_target_folder + InFileName

            Img_Input  = cv2.imread(InFile)
            Img_Input  = cv2.resize(Img_Input, (self.inputSize, self.inputSize))
            Img_Input  = Img_Input.transpose((2,0,1))
            Img_Input  = torch.from_numpy(Img_Input.astype(np.float32)/255.0)
            
            Img_Guide  = cv2.imread(GdFile)
            Img_Guide  = cv2.resize(Img_Guide, (self.guideSize, self.guideSize))
            Img_Guide  = Img_Guide.transpose((2,0,1))
            Img_Guide  = torch.from_numpy(Img_Guide.astype(np.float32)/255.0)

            Img_Target = cv2.imread(GtFile)
            Img_Target = cv2.resize(Img_Target, (self.targetSize, self.targetSize))
            Img_Target = Img_Target.transpose((2,0,1))
            Img_Target = torch.from_numpy(Img_Target.astype(np.float32)/255.0)
            
            depth_information = np.load(InDepthFile, allow_pickle=True)
            Depth_Input = depth_information.item().get('normalized_depth')
            Depth_Input = cv2.resize(Depth_Input, (self.inputSize, self.inputSize))
            Depth_Input = T.ToTensor()(Depth_Input)
            
            depth_information = np.load(GdDepthFile, allow_pickle=True)
            Depth_Guide = depth_information.item().get('normalized_depth')
            Depth_Guide = cv2.resize(Depth_Guide, (self.guideSize, self.guideSize))
            Depth_Guide = T.ToTensor()(Depth_Guide)

            InTmp  = -1
            InDir  = -1
            GdTmp  = -1
            GdDir  = -1
            GtTmp  = -1
            GtDir  = -1
            CVaild = -1
            FValid = -1
            
        if(self.mode == 'test1'):
            mode, InFileName  = self.image_filenames[index]
            InFile      = self.test1_input_folder + InFileName
            InDepthFile = self.test1_input_folder + InFileName[:8] + '.npy'
            GdFile      = self.test1_guide_folder + 'Image049_4500_E.png'
            GdDepthFile = self.test1_guide_folder + 'Image049.npy'

            Img_Input  = cv2.imread(InFile)
            Img_Input  = cv2.resize(Img_Input, (self.inputSize, self.inputSize))
            Img_Input  = Img_Input.transpose((2,0,1))
            Img_Input  = torch.from_numpy(Img_Input.astype(np.float32)/255.0)
            
            Img_Guide  = cv2.imread(GdFile)
            Img_Guide  = cv2.resize(Img_Guide, (self.guideSize, self.guideSize))
            Img_Guide  = Img_Guide.transpose((2,0,1))
            Img_Guide  = torch.from_numpy(Img_Guide.astype(np.float32)/255.0)

            depth_information = np.load(InDepthFile, allow_pickle=True)
            Depth_Input = depth_information.item().get('normalized_depth')
            Depth_Input = cv2.resize(Depth_Input, (self.inputSize, self.inputSize))
            Depth_Input = T.ToTensor()(Depth_Input)
            
            depth_information = np.load(GdDepthFile, allow_pickle=True)
            Depth_Guide = depth_information.item().get('normalized_depth')
            Depth_Guide = cv2.resize(Depth_Guide, (self.guideSize, self.guideSize))
            Depth_Guide = T.ToTensor()(Depth_Guide)

            GtFile = ''
            Img_Target = -1
            InTmp  = 4
            InDir  = 1
            GdTmp  = 2
            GdDir  = 0
            GtTmp  = -1
            GtDir  = -1
            CVaild = -1
            FValid = -1
                
        if(self.mode == 'test2'):
            mode, InFileName  = self.image_filenames[index]
            InFile      = self.test2_input_folder + InFileName
            InDepthFile = self.test2_input_folder + InFileName[:7] + '.npy'
            GdFile      = self.test2_guide_folder + InFileName
            GdDepthFile = self.test2_guide_folder + InFileName[:7] + '.npy'

            Img_Input  = cv2.imread(InFile)
            Img_Input  = cv2.resize(Img_Input, (self.inputSize, self.inputSize))
            Img_Input  = Img_Input.transpose((2,0,1))
            Img_Input  = torch.from_numpy(Img_Input.astype(np.float32)/255.0)
            
            Img_Guide  = cv2.imread(GdFile)
            Img_Guide  = cv2.resize(Img_Guide, (self.guideSize, self.guideSize))
            Img_Guide  = Img_Guide.transpose((2,0,1))
            Img_Guide  = torch.from_numpy(Img_Guide.astype(np.float32)/255.0)

            depth_information = np.load(InDepthFile, allow_pickle=True)
            Depth_Input = depth_information.item().get('normalized_depth')
            Depth_Input = cv2.resize(Depth_Input, (self.inputSize, self.inputSize))
            Depth_Input = T.ToTensor()(Depth_Input)
            
            depth_information = np.load(GdDepthFile, allow_pickle=True)
            Depth_Guide = depth_information.item().get('normalized_depth')
            Depth_Guide = cv2.resize(Depth_Guide, (self.guideSize, self.guideSize))
            Depth_Guide = T.ToTensor()(Depth_Guide)

            GtFile = ''
            Img_Target = -1
            InTmp  = -1
            InDir  = -1
            GdTmp  = -1
            GdDir  = -1
            GtTmp  = -1
            GtDir  = -1
            CVaild = -1
            FValid = -1
                
        data_dict = {'Img_Input': Img_Input, 'Depth_Input': Depth_Input, 'InTmp':InTmp, 'InDir': InDir, 'InPath': InFile, 
                     'Img_Guide': Img_Guide, 'Depth_Guide': Depth_Guide, 'GdTmp':GdTmp, 'GdDir': GdDir, 'GdPath': GdFile,
                     'Img_Target': Img_Target, 'GtTmp':GtTmp, 'GtDir': GtDir, 'GtPath': GtFile, 'CVaild': CVaild, 'FValid': FValid
                    }

        return data_dict  

    def __len__(self):
        return len(self.image_filenames)
        
import os
import cv2
import torch
import skimage
import skimage.io
import skimage.metrics
import numpy as np
import torchvision.transforms as T
from Deploy import DGateNet, DFuseNet
from Utils import count_parameters_in_MB, comp_multadds, comp_multadds_fw

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

def convert_onnx():
    opts = BaseOptions()
    opts.isTrain = False
    
    Remodel = CreateModel(opts)
    Remodel.cuda().eval()

    dummy_input1 = torch.randn(1, 3, 512, 512, device='cuda')
    dummy_input2 = torch.randn(1, 1, 512, 512, device='cuda')
    input_names  = ["intIMG", "inDepth", 'gdIMG', 'gdDepth']
    output_names = ["fakeImg", 'inLight', 'gdLight', 'inTmp', 'gdTmp']
    onnxfile     = "DFuseNet.onnx"

    output = torch.onnx.export(Remodel, (dummy_input1, dummy_input2, dummy_input1, dummy_input2), onnxfile, verbose=True, input_names=input_names, output_names=output_names)

def computeComplexity():
    Remodel = DFuseNet()
    Remodel.cuda().eval()
    
    ## compute parameters
    print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in Remodel.parameters()])))
    print('Total Params = %.2fMB' % count_parameters_in_MB(Remodel))

    mult_adds = comp_multadds(Remodel, input_size=(512, 512))
    print("compute_average_flops_cost = %.2fMB" % mult_adds)

def gen_res_t1(input_folder, Save_dir):
    Remodel1 = DFuseNet()
    Remodel1.cuda().eval()
    Remodel1.load_params('Weights/Epoch_best108.pth')

    Remodel2 = DFuseNet()
    Remodel2.cuda().eval()
    Remodel2.load_params('Weights/Epoch_best159.pth')

    InFiles = os.listdir(input_folder)
    for file in InFiles:
        if(not is_image_file(file)):
            continue 
            
        img_path  = input_folder + file
        Img_Input  = cv2.imread(img_path)
        Img_Input  = cv2.resize(Img_Input, (512, 512))
        Img_Input  = Img_Input.transpose((2,0,1))
        Img_Input  = torch.from_numpy(Img_Input.astype(np.float32)/255.0).unsqueeze(0).cuda()

        InDepthFile = input_folder + file[:8] + '.npy'
        depth_information = np.load(InDepthFile, allow_pickle=True)
        Depth_Input = depth_information.item().get('normalized_depth')
        Depth_Input = cv2.resize(Depth_Input, (512, 512))
        Depth_Input = T.ToTensor()(Depth_Input).unsqueeze(0).cuda()
        
        guide_path = 'Weights/Image014_4500_E.png'
        Img_Guide  = cv2.imread(guide_path)
        Img_Guide  = cv2.resize(Img_Guide, (512, 512))
        Img_Guide  = Img_Guide.transpose((2,0,1))
        Img_Guide  = torch.from_numpy(Img_Guide.astype(np.float32)/255.0).unsqueeze(0).cuda()

        GdDepthFile = 'Weights/Image014.npy'
        depth_information = np.load(GdDepthFile, allow_pickle=True)
        Depth_Guide = depth_information.item().get('normalized_depth')
        Depth_Guide = cv2.resize(Depth_Guide, (512, 512))
        Depth_Guide = T.ToTensor()(Depth_Guide).unsqueeze(0).cuda()
        
        fake_img1, guide_dir1, guide_tmp1 = Remodel1(Img_Input, Depth_Input, Img_Guide, Depth_Guide)
        fake_img2, guide_dir2, guide_tmp2 = Remodel2(Img_Input, Depth_Input, Img_Guide, Depth_Guide)
        fake_img = (fake_img1 + fake_img2)/2.0
        
        fake_img = fake_img[0].cpu().data.numpy()
        fake_img = fake_img.transpose((1,2,0))
        fake_img = np.squeeze(fake_img)
        fake_img = (fake_img*255.0).astype(np.uint8)
        fake_img = cv2.resize(fake_img, (1024, 1024))
        
        cv2.imwrite(os.path.join(Save_dir, file), fake_img)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
   # convert_onnx()
   # computeComplexity()
   
    Save_dir      = 'work_space/Test_Track1_DeepBlueAI'     ## Result save dir
    input_folder  = "/data/hejy/datasets/NRITE_Relight/Aug/test_t1/"   ## Change to dir of Track1 test dataset 

    gen_res_t1(input_folder, Save_dir)
    
    
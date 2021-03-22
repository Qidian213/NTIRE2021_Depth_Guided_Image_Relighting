import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
                     
class BBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batchNorm_type=0, act=None):
        super(BBlock, self).__init__()

        m = []
        m.append(default_conv(in_channels, out_channels, kernel_size))
        if(batchNorm_type == 1):
            m.append(nn.BatchNorm2d(out_channels))
        if(batchNorm_type == 2):
            m.append(nn.InstanceNorm2d(out_channels))
        if(batchNorm_type == 3):
            m.append(nn.GroupNorm(out_channels, out_channels))
        if(not act is None):
            m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, batchNorm_type=2, inplace=True):
        super(BasicBlock, self).__init__()
        self.inplanes  = inplanes
        self.outplanes = outplanes
        
        self.block1 = BBlock(inplanes, outplanes, kernel_size=3, stride=1, batchNorm_type=batchNorm_type, act=nn.ReLU(inplace))
        self.block2 = BBlock(outplanes, outplanes, kernel_size=3, stride=1, batchNorm_type=batchNorm_type, act=nn.ReLU(inplace))
        
        self.shortcuts = default_conv(inplanes, outplanes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)

        if self.inplanes != self.outplanes:
            out = out+self.shortcuts(x)
        else:
            out = out+x  
        return out
        
class Upsampler(nn.Module):
    def __init__(self, inplanes, outplanes, scale_factor=2, mode='bilinear'):
        super(Upsampler, self).__init__()
        
        self.upSample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=False)
        self.conv3x3  = default_conv(inplanes, outplanes, kernel_size=3, stride=1, bias=False)
        self.pixshuffle = nn.PixelShuffle(2)
        
    def forward(self, x):
        out = self.upSample(x)
        out = self.conv3x3(out)
        out = self.pixshuffle(out)
        return out
        
class lightingNet(nn.Module):
    def __init__(self, ncDir, ncTmp, numDir, numTmp, ncMiddle):
        super(lightingNet, self).__init__()
        self.ncDir    = ncDir
        self.ncTmp    = ncTmp
        self.ncMiddle = ncMiddle
        self.numDir   = numDir
        self.numTmp   = numTmp
        self.Is_Light = True

        self.predict_dir_FC1   = nn.Conv2d(self.ncDir,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.predict_dir_relu1 = nn.PReLU()
        self.predict_dir_FC2   = nn.Conv2d(self.ncMiddle, self.numDir, kernel_size=1, stride=1, bias=False)

        self.predict_tmp_FC1   = nn.Conv2d(self.ncTmp, self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.predict_tmp_relu1 = nn.PReLU()
        self.predict_tmp_FC2   = nn.Conv2d(self.ncMiddle, self.numTmp, kernel_size=1, stride=1, bias=False)

        self.post_dir_FC1   = nn.Conv2d(self.numDir, self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.post_dir_relu1 = nn.PReLU()
        self.post_dir_FC2   = nn.Conv2d(self.ncMiddle, self.ncDir, kernel_size=1, stride=1, bias=False)
        self.post_dir_relu2 = nn.ReLU()  

        self.post_tmp_FC1   = nn.Conv2d(self.numTmp, self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.post_tmp_relu1 = nn.PReLU()
        self.post_tmp_FC2   = nn.Conv2d(self.ncMiddle, self.ncTmp, kernel_size=1, stride=1, bias=False)
        self.post_tmp_relu2 = nn.ReLU() 

    def forward(self, innerFeat, guide_feat):
        # predict lighting  dir 
        x = innerFeat[:, 0:self.ncDir,:,:] 
        b, n, row, col = x.shape            
        feat  = x.mean(dim=(2,3), keepdim=True) 
        light = self.predict_dir_relu1(self.predict_dir_FC1(feat))
        light = self.predict_dir_FC2(light)

        gx = guide_feat[:, 0:self.ncDir,:,:] 
        b, n, row, col = gx.shape   
        gfeat  = gx.mean(dim=(2,3), keepdim=True) 
        glight = self.predict_dir_relu1(self.predict_dir_FC1(gfeat))
        glight = self.predict_dir_FC2(glight)

        # predict lighting  tmp 
        x = innerFeat[:, self.ncDir:self.ncDir+self.ncTmp,:,:] 
        b, n, row, col = x.shape            
        feat = x.mean(dim=(2,3), keepdim=True) 
        tmp = self.predict_tmp_relu1(self.predict_tmp_FC1(feat))
        tmp = self.predict_tmp_FC2(tmp)

        gx = guide_feat[:, self.ncDir:self.ncDir+self.ncTmp,:,:] 
        b, n, row, col = gx.shape   
        gfeat = gx.mean(dim=(2,3), keepdim=True) 
        gtmp = self.predict_tmp_relu1(self.predict_tmp_FC1(gfeat))
        gtmp = self.predict_tmp_FC2(gtmp)

        upFeat = self.post_dir_relu1(self.post_dir_FC1(glight))
       # upFeat = self.post_dir_relu1(self.post_dir_FC1(self.lightdir))
        upFeat = self.post_dir_relu2(self.post_dir_FC2(upFeat))
        upFeat = upFeat.repeat((1,1,row, col))
        innerFeat[:,0:self.ncDir,:,:] = upFeat
        
        upFeat = self.post_tmp_relu1(self.post_tmp_FC1(gtmp))
       # upFeat = self.post_tmp_relu1(self.post_tmp_FC1(self.lighttmp))
        upFeat = self.post_tmp_relu2(self.post_tmp_FC2(upFeat))
        upFeat = upFeat.repeat((1,1,row, col))
        innerFeat[:,self.ncDir:self.ncDir+self.ncTmp,:,:] = upFeat
       
        return innerFeat, light.view(b,-1), glight.view(b,-1), tmp.view(b,-1), gtmp.view(b,-1)

class BFuseNet(nn.Module):
    def __init__(self, ):
        super(BFuseNet, self).__init__()
        self.inChannel   = 3
        self.baseFilter  = 48
        self.ncDir       = self.baseFilter*12
        self.ncTmp       = self.baseFilter*4
        self.DirLight    = 8
        self.TmpLight    = 5
        
        self.ncHG0 = 2*self.baseFilter    # 96
        self.ncHG1 = 4*self.baseFilter    # 192
        self.ncHG2 = 8*self.baseFilter    # 384
        self.ncHG3 = 16*self.baseFilter   # 768
        
        self.pre_block = BasicBlock(self.inChannel, self.baseFilter, batchNorm_type=1)        # 3->48
        self.depth_pre_block = BasicBlock(1, self.baseFilter, batchNorm_type=1)  # 1->48
        
        self.HG0_upper      = BasicBlock(self.baseFilter, self.baseFilter, batchNorm_type=1)  # 48->48
        self.HG0_downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.HG0_low1       = BasicBlock(self.baseFilter, self.ncHG0, batchNorm_type=1)       # 48->96
        self.HG0_low1_depth = BasicBlock(self.baseFilter, self.ncHG0, batchNorm_type=1)       # 48->96
        self.HG0_low2       = BasicBlock(self.ncHG0*2, self.baseFilter, batchNorm_type=2)     # 192->48
        self.HG0_upSample   = nn.Upsample(scale_factor=2, mode='nearest')

        self.HG1_upper      = BasicBlock(self.ncHG0, self.ncHG0, batchNorm_type=1)            # 96->96
        self.HG1_downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.HG1_low1       = BasicBlock(self.ncHG0, self.ncHG1, batchNorm_type=1)            # 96->192
        self.HG1_low1_depth = BasicBlock(self.ncHG0, self.ncHG1, batchNorm_type=1)            # 96->192
        self.HG1_low2       = BasicBlock(self.ncHG1*2, self.ncHG0, batchNorm_type=2)          # 384->96
        self.HG1_upSample   = nn.Upsample(scale_factor=2, mode='nearest')
 
        self.HG2_upper      = BasicBlock(self.ncHG1, self.ncHG1, batchNorm_type=1)            # 192->192
        self.HG2_downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.HG2_low1       = BasicBlock(self.ncHG1, self.ncHG2, batchNorm_type=1)            # 192->384
        self.HG2_low1_depth = BasicBlock(self.ncHG1, self.ncHG2, batchNorm_type=1)            # 192->384
        self.HG2_low2       = BasicBlock(self.ncHG2*2, self.ncHG1, batchNorm_type=2)          # 768->192
        self.HG2_upSample   = nn.Upsample(scale_factor=2, mode='nearest')

        self.HG3_upper      = BasicBlock(self.ncHG2, self.ncHG2, batchNorm_type=1)            # 383->384
        self.HG3_downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.HG3_low1       = BasicBlock(self.ncHG2, self.ncHG3, batchNorm_type=1)            # 384->768
        self.HG3_low1_depth = BasicBlock(self.ncHG2, self.ncHG3, batchNorm_type=1)            # 384->768
        self.HG3_low2       = BasicBlock(self.ncHG3, self.ncHG2, batchNorm_type=2)            # 768->384
        self.HG3_upSample   = nn.Upsample(scale_factor=2, mode='nearest')

        self.lightnet = lightingNet(self.ncDir, self.ncTmp, self.DirLight, self.TmpLight, 256) 

        self.final_block = BasicBlock(self.baseFilter*2, self.baseFilter, batchNorm_type=1)
        self.out_layer   = nn.Conv2d(self.baseFilter, 3, kernel_size=1, stride=1, padding=0)

       # self.init_weights()

    def forward(self, input_img, input_depth, guide_img, guide_depth): 
        ### guide image
        feat_guide  = self.pre_block(guide_img)
        depth_guide = self.depth_pre_block(guide_depth)
        
        feat_guide  = self.HG0_downSample(feat_guide+depth_guide)
        depth_guide = self.HG0_downSample(depth_guide)
        feat_guide  = self.HG0_low1(feat_guide)
        depth_guide = self.HG0_low1_depth(depth_guide)
        
        feat_guide  = self.HG1_downSample(feat_guide+depth_guide)
        depth_guide = self.HG1_downSample(depth_guide)
        feat_guide  = self.HG1_low1(feat_guide)
        depth_guide = self.HG1_low1_depth(depth_guide)
        
        feat_guide  = self.HG2_downSample(feat_guide+depth_guide)
        depth_guide = self.HG2_downSample(depth_guide)
        feat_guide  = self.HG2_low1(feat_guide)
        depth_guide = self.HG2_low1_depth(depth_guide)
        
        feat_guide  = self.HG3_downSample(feat_guide+depth_guide)
        depth_guide = self.HG3_downSample(depth_guide)
        feat_guide  = self.HG3_low1(feat_guide)
        depth_guide = self.HG3_low1_depth(depth_guide)
        
        ### input image
        feat_input  = self.pre_block(input_img)
        depth_input = self.depth_pre_block(input_depth)
        
        feat0_upper  = self.HG0_upper(feat_input)
        feat_input   = self.HG0_downSample(feat_input+depth_input)
        depth_input  = self.HG0_downSample(depth_input)
        feat_input   = self.HG0_low1(feat_input)
        depth_input  = self.HG0_low1_depth(depth_input)
        
        feat1_upper = self.HG1_upper(feat_input)
        feat_input  = self.HG1_downSample(feat_input+depth_input)
        depth_input = self.HG1_downSample(depth_input)
        feat_input  = self.HG1_low1(feat_input)
        depth_input = self.HG1_low1_depth(depth_input)
        
        feat2_upper = self.HG2_upper(feat_input)
        feat_input  = self.HG2_downSample(feat_input+depth_input)
        depth_input = self.HG2_downSample(depth_input)
        feat_input  = self.HG2_low1(feat_input)
        depth_input = self.HG2_low1_depth(depth_input)
        
        feat3_upper = self.HG3_upper(feat_input)
        feat_input  = self.HG3_downSample(feat_input+depth_input)
        depth_input = self.HG3_downSample(depth_input)
        feat_input  = self.HG3_low1(feat_input)
        depth_input = self.HG3_low1_depth(depth_input)

        light_feat, input_dir, guide_dir, input_tmp, guide_tmp = self.lightnet(feat_input+depth_input, feat_guide+depth_guide)
        
        light_feat = self.HG3_low2(light_feat)
        light_feat = self.HG3_upSample(light_feat)
        light_feat = torch.cat((light_feat, feat3_upper), dim=1)

        light_feat = self.HG2_low2(light_feat)
        light_feat = self.HG2_upSample(light_feat)
        light_feat = torch.cat((light_feat, feat2_upper), dim=1)
        
        light_feat = self.HG1_low2(light_feat)
        light_feat = self.HG1_upSample(light_feat)
        light_feat = torch.cat((light_feat, feat1_upper), dim=1)
        
        light_feat = self.HG0_low2(light_feat)
        light_feat = self.HG0_upSample(light_feat)
        light_feat = torch.cat((light_feat, feat0_upper), dim=1)
        
        light_feat = self.final_block(light_feat)
        light_feat = self.out_layer(light_feat)
        out_img    = torch.sigmoid(light_feat)
        
        return out_img, input_dir, guide_dir, input_tmp, guide_tmp

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight.data, gain=0.02)
                # if m.bias is not None:
                    # nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def load_params(self, file):
        if(file == '' or file == None):
            return 
        
        checkpoint = torch.load(file)
               
        if('state_dict' in checkpoint.keys()):
            checkpoint = checkpoint['state_dict']
        
        model_state_dict = self.state_dict()
        new_state_dict   = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            if name in model_state_dict:
                if v.shape != model_state_dict[name].shape:
                    print('Skip loading parameter {}, required shape{}, '\
                          'loaded shape{}.'.format(name, model_state_dict[name].shape, v.shape))
                    new_state_dict[name] = model_state_dict[name]
            else:
                print('Drop parameter {}.'.format(name))
            
        for key in model_state_dict.keys():
            if(key not in new_state_dict.keys()):
                print('No param {}.'.format(key))
                new_state_dict[key] = model_state_dict[key]
            
        self.load_state_dict(new_state_dict, strict=False)
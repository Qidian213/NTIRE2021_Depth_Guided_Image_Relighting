import os
import cv2
import glob
import skimage
import skimage.io
import skimage.metrics
import numpy as np
import torch
import shutil
from tensorboardX import SummaryWriter
from Options import BaseOptions
from Dataset import CreateDataLoader
from Models  import CreateModel
from Optimizers import Get_Optimizer
from Losses import Get_LossFunction
from Utils import GetLogger, Get_Time_Stamp, AverageMeter, Accuracy

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
torch.backends.cudnn.benchmark = True

def Setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class Mainer(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        
        self.train_dataloader, self.T1_Val_dataloader, self.T2_Val_dataloader, self.T1_Test_dataloader, self.T2_Test_dataloader= CreateDataLoader(self.cfgs)
        self.epoch_batchs = len(self.train_dataloader)

        self.model = CreateModel(self.cfgs)
        self.model.cuda()
        if(self.cfgs.preTrained != ''):
            self.model.load_params(self.cfgs.preTrained)
        self.model = torch.nn.DataParallel(self.model)

        self.optim_schedulers = Get_Optimizer(self.cfgs, self.model, self.epoch_batchs)
        self.loss_meter       = Get_LossFunction(self.cfgs)

        self.writer = SummaryWriter(self.cfgs.Save_dir)
        shutil.copyfile('Options/options.py', self.cfgs.Save_dir + '/config.py')
        shutil.copyfile('Train_Guide.py', self.cfgs.Save_dir + '/Train_Guide.py')
        
    def train(self,):
        best_PSNR  = 0
        best_SSIM  = 0
        best_epoch = 0
        for epoch in range(self.cfgs.max_epoch):
            self.train_epoch(epoch)

            if(epoch >0 and epoch %3 ==0):
                self.T1_Test(epoch)
                self.T2_Test(epoch)
                PSNR_T1, SSIM_T1 = self.T1_Val(epoch)
                PSNR_T2, SSIM_T2 = self.T2_Val(epoch)
                SSIM = (SSIM_T2 + SSIM_T1)/2.0
                PSNR = (PSNR_T2 + PSNR_T1)/2.0

                if(SSIM > best_SSIM):
                    best_SSIM  = SSIM
                    best_PSNR  = PSNR
                    best_epoch = epoch
                    save_file  = os.path.join(self.cfgs.Save_dir, 'Model/Epoch_best.pth')
                    torch.save(self.model.state_dict(),save_file)
                    files = glob.glob(self.cfgs.Save_dir + '/T2/Pair*.png')
                    for file in files:  
                        shutil.copyfile(file, file.replace('/T2/', '/T2/B_'))
                    files = glob.glob(self.cfgs.Save_dir + '/T1/Image*.png')
                    for file in files:  
                        shutil.copyfile(file, file.replace('/T1/', '/T1/B_'))

            if(epoch >20 and epoch %2 ==0):        
                save_file = os.path.join(self.cfgs.Save_dir, f'Model/Epoch_{epoch}.pth')
                torch.save(self.model.state_dict(),save_file)
            
            logger.info(f"Best_Epoch: {best_epoch}, best_PSNR: {best_PSNR:.5f}, best_SSIM: {best_SSIM:.5f} \r")

    def train_epoch(self, epoch):
        log_Loss   = AverageMeter()
        log_SSIM   = AverageMeter()
        log_PSNR   = AverageMeter()
        log_DirAcc = AverageMeter()
        log_TmpAcc = AverageMeter()
        
        self.model.train()
        for step, data in enumerate(self.train_dataloader):
            Img_Input   = data['Img_Input'].cuda()
            Depth_Input = data['Depth_Input'].cuda()
            InTmp       = data['InTmp'].cuda()
            InDir       = data['InDir'].cuda()

            Img_Guide   = data['Img_Guide'].cuda()
            Depth_Guide = data['Depth_Guide'].cuda()
            GdTmp       = data['GdTmp'].cuda() 
            GdDir       = data['GdDir'].cuda()

            Img_Target = data['Img_Target'].cuda()
            GtTmp      = data['GtTmp'].cuda() 
            GtDir      = data['GtDir'].cuda()
     
            CVaild     = data['CVaild'].cuda()
            FValid     = data['FValid'].cuda()

            fake_img, pinlight, pgdlight, pinTmp, pgdTmp = self.model(Img_Input, Depth_Input, Img_Guide, Depth_Guide)

            loss_L1    = self.loss_meter['criterionMSE'](fake_img, Img_Target,FValid)
            loss_SSIM  = self.loss_meter['criterionSSIM'](fake_img, Img_Target,FValid)
            if(self.cfgs.useGrad):
                loss_Grad  = self.loss_meter['criterionGrad'](fake_img, Img_Target,FValid)
            if(self.cfgs.useLPIPS):   
                loss_LPIPS = self.loss_meter['criterionLPIPS'](fake_img, Img_Target,FValid)

            loss_input_dir = self.loss_meter['criterionFDIR'](pinlight, InDir, CVaild)
            loss_guide_dir = self.loss_meter['criterionFDIR'](pgdlight, GdDir, CVaild)
            loss_input_tmp = self.loss_meter['criterionFTMP'](pinTmp, InTmp, CVaild)
            loss_guide_tmp = self.loss_meter['criterionFTMP'](pgdTmp, GdTmp, CVaild)
            
            loss = loss_L1 + 1.0*(1-loss_SSIM) + loss_input_dir + loss_guide_dir + loss_input_tmp + loss_guide_tmp
            
            if(self.cfgs.useLPIPS): 
                loss += 0.01*loss_LPIPS.mean()
            if(self.cfgs.useGrad):
                loss += loss_Grad
                
            self.optim_schedulers['optimizer'].zero_grad()
            loss.backward()
            self.optim_schedulers['optimizer'].step()

            Indir_acc1 = Accuracy(pinlight, InDir, topk=(1,))
            Gddir_acc1 = Accuracy(pgdlight, GdDir, topk=(1,))
            Intmp_acc1 = Accuracy(pinTmp, InTmp, topk=(1,))
            Gdtmp_acc1 = Accuracy(pgdTmp, GdTmp, topk=(1,))
            log_DirAcc.update((Indir_acc1[0].item() + Gddir_acc1[0].item())/2,len(GtDir))
            log_TmpAcc.update((Intmp_acc1[0].item() + Gdtmp_acc1[0].item())/2,len(GtDir))
            
            log_Loss.update(loss.item(), self.cfgs.batchSize)
            log_SSIM.update(loss_SSIM.item(), self.cfgs.batchSize)
        #    log_PSNR.update(loss_PSNR.item(), self.cfgs.batchSize)
            
            if step%50 == 0:            
                self.writer.add_scalar('log_Loss:', log_Loss.avg, self.epoch_batchs*epoch + step//10)
                self.writer.add_scalar('log_SSIM:', log_SSIM.avg, self.epoch_batchs*epoch + step//10)
                logger.info(f"iter: {step}/{self.epoch_batchs}/{epoch}, log_Loss: {log_Loss.avg:.5f}, log_DirAcc: {log_DirAcc.avg:.5f}, " 
                            f"log_TmpAcc: {log_TmpAcc.avg:.5f}, loss_SSIM: {log_SSIM.avg:.5f}, log_PSNR: {log_PSNR.avg:.5f}, LR: {'%e'%self.optim_schedulers['optimizer'].param_groups[0]['lr']}\r")
        
        self.optim_schedulers['scheduler'].step()

    def T1_Val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            Rc_Lists = []
            TargetPaths = []
            
            for step, data in enumerate(self.T1_Val_dataloader):
                Depth_Input = data['Depth_Input'].cuda()
                Img_Input   = data['Img_Input'].cuda()
                InTmp       = data['InTmp'].cuda()
                InDir       = data['InDir'].cuda()

                Depth_Guide = data['Depth_Guide'].cuda()
                Img_Guide   = data['Img_Guide'].cuda()
                GdTmp       = data['GdTmp'].cuda() 
                GdDir       = data['GdDir'].cuda()

                Img_Target  = data['Img_Target'].cuda()
                
                GtPath = data['GtPath'][0]
                TargetPaths.append(GtPath)
                
                fake_img, _, _, _, _ = self.model(Img_Input, Depth_Input, Img_Guide, Depth_Guide)

                fake_img = fake_img[0].cpu().data.numpy()
                fake_img = fake_img.transpose((1,2,0))
                fake_img = np.squeeze(fake_img)
                fake_img = (fake_img*255.0).astype(np.uint8)
                fake_img = cv2.resize(fake_img, (1024, 1024))

                Rc_Lists.append(fake_img)
                cv2.imwrite(os.path.join(self.cfgs.Save_dir, 'V1/'+GtPath.split('/')[-1]), fake_img)

            PSNRs = []
            SSIMs = []
            for rc, gtpath in zip(Rc_Lists, TargetPaths):
                gt = cv2.imread(gtpath)
                gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
                rc = cv2.cvtColor(rc,cv2.COLOR_BGR2RGB)
                PSNRs.append(skimage.metrics.peak_signal_noise_ratio(gt, rc))
                SSIMs.append(skimage.metrics.structural_similarity(gt, rc, multichannel=True))
            PSNR = np.mean(PSNRs)
            SSIM = np.mean(SSIMs)
            
            self.Rc_Lists = []
            logger.info(f"T1--Epoch: {epoch}, PSNR: {PSNR:.5f}, SSIM: {SSIM:.5f} \r")
            return PSNR, SSIM
            
    def T2_Val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            Rc_Lists = []
            TargetPaths = []
            
            for step, data in enumerate(self.T2_Val_dataloader):
                Depth_Input = data['Depth_Input'].cuda()
                Img_Input   = data['Img_Input'].cuda()
                InTmp       = data['InTmp'].cuda()
                InDir       = data['InDir'].cuda()

                Depth_Guide = data['Depth_Guide'].cuda()
                Img_Guide   = data['Img_Guide'].cuda()
                GdTmp       = data['GdTmp'].cuda() 
                GdDir       = data['GdDir'].cuda()

                Img_Target  = data['Img_Target'].cuda()
                
                GtPath = data['GtPath'][0]
                TargetPaths.append(GtPath)
                
                fake_img, _, _, _, _ = self.model(Img_Input, Depth_Input, Img_Guide, Depth_Guide)

                fake_img = fake_img[0].cpu().data.numpy()
                fake_img = fake_img.transpose((1,2,0))
                fake_img = np.squeeze(fake_img)
                fake_img = (fake_img*255.0).astype(np.uint8)
                fake_img = cv2.resize(fake_img, (512, 512))

                Rc_Lists.append(fake_img)
                cv2.imwrite(os.path.join(self.cfgs.Save_dir, 'V2/'+GtPath.split('/')[-1]), fake_img)

            PSNRs = []
            SSIMs = []
            for rc, gtpath in zip(Rc_Lists, TargetPaths):
                gt = cv2.imread(gtpath)
                gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
                rc = cv2.cvtColor(rc,cv2.COLOR_BGR2RGB)
                PSNRs.append(skimage.metrics.peak_signal_noise_ratio(gt, rc))
                SSIMs.append(skimage.metrics.structural_similarity(gt, rc, multichannel=True))
            PSNR = np.mean(PSNRs)
            SSIM = np.mean(SSIMs)
            
            self.Rc_Lists = []
            logger.info(f"T2--Epoch: {epoch}, PSNR: {PSNR:.5f}, SSIM: {SSIM:.5f} \r")
            return PSNR, SSIM

    def T1_Test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.T1_Test_dataloader):
                Depth_Input = data['Depth_Input'].cuda()
                Img_Input   = data['Img_Input'].cuda()
                InTmp       = data['InTmp'].cuda()
                InDir       = data['InDir'].cuda()

                Depth_Guide = data['Depth_Guide'].cuda()
                Img_Guide   = data['Img_Guide'].cuda()
                GdTmp       = data['GdTmp'].cuda() 
                GdDir       = data['GdDir'].cuda()

                InFile = data['InPath'][0]
                
                fake_img, _, _, _, _ = self.model(Img_Input, Depth_Input, Img_Guide, Depth_Guide)

                fake_img = fake_img[0].cpu().data.numpy()
                fake_img = fake_img.transpose((1,2,0))
                fake_img = np.squeeze(fake_img)
                fake_img = (fake_img*255.0).astype(np.uint8)
                fake_img = cv2.resize(fake_img, (1024, 1024))

                cv2.imwrite(os.path.join(self.cfgs.Save_dir, 'T1/'+InFile.split('/')[-1]), fake_img)

            logger.info(f"T1--Epoch: {epoch}, Test \r")

    def T2_Test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.T2_Test_dataloader):
                Depth_Input = data['Depth_Input'].cuda()
                Img_Input   = data['Img_Input'].cuda()
                InTmp       = data['InTmp'].cuda()
                InDir       = data['InDir'].cuda()

                Depth_Guide = data['Depth_Guide'].cuda()
                Img_Guide   = data['Img_Guide'].cuda()
                GdTmp       = data['GdTmp'].cuda() 
                GdDir       = data['GdDir'].cuda()

                InFile = data['InPath'][0]
                
                fake_img, _, _, _, _ = self.model(Img_Input, Depth_Input, Img_Guide, Depth_Guide)

                fake_img = fake_img[0].cpu().data.numpy()
                fake_img = fake_img.transpose((1,2,0))
                fake_img = np.squeeze(fake_img)
                fake_img = (fake_img*255.0).astype(np.uint8)
                fake_img = cv2.resize(fake_img, (1024, 1024))

                cv2.imwrite(os.path.join(self.cfgs.Save_dir, 'T2/'+InFile.split('/')[-1]), fake_img)

            logger.info(f"T2--Epoch: {epoch}, Test \r")

if __name__ == '__main__':
 #   Setup_seed(233)
    train_opts = BaseOptions()
    
    train_opts.Save_dir = os.path.join(train_opts.Save_dir, train_opts.model +'_'+Get_Time_Stamp())
    if(not os.path.exists(train_opts.Save_dir)):
        os.makedirs(train_opts.Save_dir)
    if(not os.path.exists(train_opts.Save_dir + '/T1')):
        os.makedirs(train_opts.Save_dir + '/T1')
    if(not os.path.exists(train_opts.Save_dir + '/T2')):
        os.makedirs(train_opts.Save_dir + '/T2')
    if(not os.path.exists(train_opts.Save_dir + '/V1')):
        os.makedirs(train_opts.Save_dir + '/V1')
    if(not os.path.exists(train_opts.Save_dir + '/V2')):
        os.makedirs(train_opts.Save_dir + '/V2')
    if(not os.path.exists(train_opts.Save_dir + '/Model')):
        os.makedirs(train_opts.Save_dir + '/Model')
        
    logger = GetLogger("Relight" , train_opts.Save_dir + '/' + train_opts.model +'.log')
    logger.info("startup... \r")
    
    mainer = Mainer(train_opts)
    mainer.train()


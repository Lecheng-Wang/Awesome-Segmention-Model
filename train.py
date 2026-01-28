# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2026/1/11  21:08
# @Function   : 
# @Description: train file


import os
import csv
import random
import argparse
import torch
import torch.nn             as nn
import numpy                as np
import torch.optim          as optim
import torch.backends.cudnn as cudnn


from nets.unet                import Unet
from nets.deeplabv3_plus      import DeepLab
from nets.ENet                import ENet
from nets.fcn                 import FCN16s,FCN8s,FCN32s
from nets.hrnet               import hrnet
from nets.pspnet              import PSPNet
from nets.segformer           import Segformer
from nets.segnet              import SegNet
from nets.SETR                import SETR
from nets.refinenet           import rf50
from nets.UperNet             import UperNet
from nets.segnext             import SegNeXt
from nets.hrnet_ocr           import hrnetocr
from nets.mask2former         import Mask2Former

from torch.utils.data         import DataLoader
from utils.dataset            import Labeled_Model_Dataset
from utils.metrics            import Evaluator
from utils.weight_init        import weights_init
from utils.focal              import FocalLoss
from tqdm                     import tqdm


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For muti-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed(42)  # Set seed at the beginning

parser = argparse.ArgumentParser(description="Unet/DeeplabV3+/PSPNet/HRNet/SegNet/FCN/Segformer/SETR based on multi_backbone and multi_attention")
parser.add_argument('--DATASET_PATH',      type=str,   default='./datasets/')
parser.add_argument('--MODE',              type=str,   default='all', choices=['single','multi','all'])
parser.add_argument('--GPU_ID',            type=int,   default=0)
parser.add_argument('--GPU_LIST',          type=str,   default='1,2')
parser.add_argument('--USE_CPU',           type=bool,  default=False)
parser.add_argument('--BANDS',             type=int,   default=10)
parser.add_argument('--NUM_CLASS',         type=int,   default=2+1)
parser.add_argument('--LR_STEP',           type=int,   default=1)
parser.add_argument('--STEP_RATIO',        type=float, default=0.94)
parser.add_argument('--INIT_LR',           type=float, default=1e-3)
parser.add_argument('--MOMENTUM',          type=float, default=0.9)
parser.add_argument('--WEIGHT_DECAY',      type=float, default=1e-4)
parser.add_argument('--BATCH_SIZE',        type=int,   default=4)
parser.add_argument('--START_EPOCH',       type=int,   default=1)
parser.add_argument('--EPOCHS',            type=int,   default=1)
parser.add_argument('--PRETRAIN_MODEL',    type=str,   default=None)
parser.add_argument('--MODEL_SAVE_EPOCHS', type=int,   default=1)
parser.add_argument('--LOSS_TYPE',         type=str,   default='ce',       choices=['ce','focal'])
parser.add_argument('--OPTIMIZER_TYPE',    type=str,   default='adam',      choices=['adam','sgd'])
parser.add_argument('--LR_SCHEDULER',      type=str,   default='poly',     choices=['poly','step', 'cos','exp'])
parser.add_argument('--MODEL_TYPE',        type=str,   default='ocrnet',  choices=['unet','deeplab','enet','pspnet','hrnet','segnet','refinenet','fcn','segformer','setr','upernet','ocrnet','mask2former','segnext']) 
parser.add_argument('--BACKBONE_TYPE',     type=str,   default='resnet50')       # unet:vgg11/13/16/19、resnet18/34/50/101/152   deeplab:xception/mobilenet/resnet/vggnet/inception
parser.add_argument('--ATTENTION_TYPE',    type=str,   default=None,       choices=['senet','ecanet','cbam','vit','self_atten'])
parser.add_argument('--INIT_TYPE',         type=str,   default='kaiming',  choices=['kaiming','normal','xavier','orthogonal'])
args = parser.parse_args()


# Parallel Mode->
#        single: Training on only-one GPU
#        muti:  Parallel training on multiple GPUs in GPU_LIST
#        all:    Parallel training on all GPUs on this platform
if args.USE_CPU:
    device    = torch.device('cpu')
    print("The code is currently specified to use CPU for training\n", flush=True)

elif args.MODE=='single':
    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Single-GPU training mode will be used on {device}\n", flush=True)

elif args.MODE=='multi':
    args.GPU_LIST = [int(gpu_id) for gpu_id in args.GPU_LIST.split(',')]
    if torch.cuda.is_available() and all(gpu < torch.cuda.device_count() for gpu in args.GPU_LIST):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.GPU_LIST))
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Multi-GPU training mode will be used on devices {args.GPU_LIST}\n", flush=True)

    else:
        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"The current platform does not support multi-GPU mode, will use {device} for training by default\n", flush=True)

elif args.MODE=='all':
    device    = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    print(f"All {gpu_count} GPU devices on the current platform will be used for training\n", flush=True)



def main ():        
    # 1.Selecting model type
    if args.MODEL_TYPE == 'unet':
        model = Unet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE, atten_type=args.ATTENTION_TYPE)
    elif args.MODEL_TYPE == 'deeplab':
        model = DeepLab(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE, atten_type=args.ATTENTION_TYPE)
    elif args.MODEL_TYPE == 'fcn':
        model = FCN16s(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'hrnet':
        model = hrnet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=18, version='v2')                    # 18, 32, 48
    elif args.MODEL_TYPE == 'pspnet':
        model = PSPNet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE, downsample_factor=8)
    elif args.MODEL_TYPE == 'refinenet':
        model = rf50(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'enet':
        model = ENet(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'segformer':
        model = Segformer(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone='b0')                            # b0,b1,b2,b3,b4,b5
    elif args.MODEL_TYPE == 'setr':
        model = SETR(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone='Base', img_size=256)                 # Base,Large
    elif args.MODEL_TYPE == 'segnet':
        model = SegNet(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'upernet':
        model = UperNet(bands=args.BANDS, num_classes=args.NUM_CLASS) 
    elif args.MODEL_TYPE == 'segnext':
        model = SegNeXt(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone='T')                               # T,S,B,L
    elif args.MODEL_TYPE == 'ocrnet':
        model = hrnetocr(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=48) 
    elif args.MODEL_TYPE == 'mask2former':                                                                        # Error, need revision
        model = Mask2Former(bands=args.BANDS, num_classes=args.NUM_CLASS)
    else:
        raise NotImplementedError('model type [%s] is not implemented, unet/deeplab/pspnet/hrnet/segnet/fcn/segformer/setr is supported!' %args.MODEL_TYPE)
    
    # 2.Creat empty file-fold to save model file 
    os.makedirs('pth_files', exist_ok=True)

    # 3.Clearing GPU cache memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory Cleared: Allocated {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
    
    # 4.Loading pretrain model weight(if exit)
    if args.PRETRAIN_MODEL:
        if os.path.isfile(args.PRETRAIN_MODEL):
            print(f"=> Loading pretrained model from {args.PRETRAIN_MODEL}")
            checkpoint = torch.load(args.PRETRAIN_MODEL, map_location='cpu')            
            pretrained_dict = checkpoint
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            model.load_state_dict(pretrained_dict, strict=False)
            print("=> Loaded pretrained weights (strict=False)")
    else:
        print(f"=> No pretrained model found at '{args.PRETRAIN_MODEL}'")
        weights_init(model, init_type=args.INIT_TYPE)

    if args.MODE in ['multi','all'] and (not args.USE_CPU):
        model = nn.DataParallel(model)
    model = model.to(device)


    print(f"Information:\n|model:{args.MODEL_TYPE}\n|backbone:{args.BACKBONE_TYPE}\n|optimizer:{args.OPTIMIZER_TYPE}\n|batchsize:{args.BATCH_SIZE}\n|loss type:{args.LOSS_TYPE}\n|init lr:{args.INIT_LR}\n|lr scheduler:{args.LR_SCHEDULER}\n|weight decay:{args.WEIGHT_DECAY}\n|training epochs:{args.EPOCHS}\n|init type:{args.INIT_TYPE}.\n")
    # 5.Setting class-balanced weight 
    weight    = np.array([0.03247, 0.25926, 0.70827], np.float32)
    weight    = torch.from_numpy(weight.astype(np.float32)).to(device)

    # 6.Setting Loss function
    if args.LOSS_TYPE=='ce':
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1, reduction='mean')
    elif args.LOSS_TYPE=='focal':
        criterion = FocalLoss(alpha=weight, gamma=2.0, ignore_index=-1, reduction='mean')
    else:
        raise NotImplementedError('loss type [%s] is not implemented,ce/focal is supported!' %args.LOSS_TYPE)
        
    # 7.Setting Optimizer
    if args.OPTIMIZER_TYPE == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.INIT_LR, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
    elif args.OPTIMIZER_TYPE == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.INIT_LR, weight_decay=args.WEIGHT_DECAY)
    else:
        raise NotImplementedError('optimizer type [%s] is not implemented,sgd and adam is supported!' %args.OPTIMIZER_TYPE)

    # 8.Setting learning rate scheduler type
    if args.LR_SCHEDULER   == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.LR_STEP, gamma=args.STEP_RATIO)
    elif args.LR_SCHEDULER == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.STEP_RATIO)
    elif args.LR_SCHEDULER == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS)
    elif args.LR_SCHEDULER == 'poly':
        lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.EPOCHS, power=2)
    else:
        raise NotImplementedError('lr scheduler type [%s] is not implemented,cos,step,poly,exp is supported!' %args.LR_SCHEDULER)

    # 9.Reading datasets list
    with open(os.path.join(args.DATASET_PATH, "annotations/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.DATASET_PATH, "annotations/val.txt"),"r") as f:
        test_lines  = f.readlines()

    # 10.Reading datasets files in the list,and Packing datasets into dataloader
    train_datasets  = Labeled_Model_Dataset(train_lines, args.DATASET_PATH)
    test_datasets   = Labeled_Model_Dataset(test_lines,  args.DATASET_PATH)
    train_loader    = DataLoader(train_datasets,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0,pin_memory=True,drop_last=True)
    test_loader     = DataLoader(test_datasets, shuffle=False,batch_size=args.BATCH_SIZE,num_workers=0,pin_memory=True,drop_last=False)

    cudnn.benchmark = True

    # 11.Define a training object(trainer)
    trainer = Trainer(args, model, criterion, optimizer, train_loader, test_loader)

    print('Starting Epoch:', trainer.args.START_EPOCH)
    print('Total Epoches:',  trainer.args.EPOCHS)

    # 12.Creating initial CSV file to record training and evaluating results  
    with open(f'{args.MODEL_TYPE}_training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','val_loss','Acc','Kappa','mIoU','mIoU0','mIoU1','mIoU2','FWIoU','Precision','Precision0','Precision1','Precision2','Recall','Recall0','Recall1','Recall2','F1_score','F1_score0','F1_score1','F1_score2','F2_score','F2_score0','F2_score1','F2_score2'])

    # 13.Beginning to train and evaluate model(calling training method and validation methods in trainer object)
    print(f"Start training on device:{device}...")
    for epoch in range(trainer.args.EPOCHS):
        train_loss = trainer.training(epoch)
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        print("Current learning rate is:", current_lr)
        print("Training over.\n")

        print(f"Start validating on device:{device}......")
        val_loss,Acc,Kappa,mIoU,mIoU0,mIoU1,mIoU2,FWIoU,Precision,Precision0,Precision1,Precision2,Recall,Recall0,Recall1,Recall2,F1_score,F1_score0,F1_score1,F1_score2,F2_score,F2_score0,F2_score1,F2_score2=trainer.validation(epoch)
        print("Validating over.\n")

        # 13.Saving model pth-format file every periodically
        if (epoch + 1) % args.MODEL_SAVE_EPOCHS == 0:
            torch.save(model.state_dict(),'pth_files/%s-epoch%d-loss%.3f-val_loss%.3f.pth'%(args.MODEL_TYPE,(epoch+1),train_loss,val_loss))
        
        # 14.Writting training log(metrics) into recording file
        with open(f'{args.MODEL_TYPE}_training_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1,train_loss,val_loss,Acc,Kappa,mIoU,mIoU0,mIoU1,mIoU2,FWIoU,Precision,Precision0,Precision1,Precision2,Recall,Recall0,Recall1,Recall2,F1_score,F1_score0,F1_score1,F1_score2,F2_score,F2_score0,F2_score1,F2_score2])

class Trainer(object):
    def __init__(self,args,model,criterion,optimizer,train_loader,val_loader):
        self.args         = args
        self.model        = model
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.evaluator    = Evaluator(self.args.NUM_CLASS)

    def training(self, epoch):
        self.model.train()
        train_loss   = 0.0
        train_loader = tqdm(self.train_loader)
        num_batch    = len(self.train_loader)

        for i, data in enumerate(train_loader):
            img, lbl = data

            img      = img.float().to(device)
            lbl      = lbl.long().to(device)

            output   = self.model(img).float()
            loss     = self.criterion(output, lbl)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss  += loss.item()
            train_loader.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        print('Epoch: %d, numImages: %5d' % (epoch+1, num_batch * self.args.BATCH_SIZE))
        print('Train Loss: %.3f' % (train_loss / num_batch))
        return train_loss/num_batch

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        val_loss    = 0.0
        val_loader  = tqdm(self.val_loader)
        num_batch   = len(self.val_loader)
        
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                image, label = sample
                image        = image.float().to(device)
                label        = label.long().to(device)

                output       = self.model(image).float()
                loss         = self.criterion(output, label)

                val_loss     = val_loss + loss.item()
                val_loader.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
                pred         = output.data.cpu().numpy()
                label        = label.cpu().numpy()
                pred         = np.argmax(pred, axis=1)
                self.evaluator.add_batch(label, pred)

        Acc                              = self.evaluator.OverAll_Accuracy()
        Kappa                            = self.evaluator.Kappa()

        mIoU,    IoU                     = self.evaluator.mean_Intersection_over_Union()
        mIoU0,     mIoU1,     mIoU2      = IoU

        FWIoU                            = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        mPrecision,Precision             = self.evaluator.Precision()
        Precision0,Precision1,Precision2 = Precision

        mRecall,   Recall                = self.evaluator.Recall()
        Recall0,   Recall1,   Recall2    = Recall

        mF1_score,F1_score               = self.evaluator.F1_Score()
        F1_score0, F1_score1, F1_score2  = F1_score

        mF2_score,F2_score               = self.evaluator.F2_Score()
        F2_score0, F2_score1, F2_score2  = F2_score

        print('Validation Result:')
        print('Epoch:%d, numImages: %5d' % (epoch+1, num_batch * self.args.BATCH_SIZE))
        print("Epoch:{}, Acc:{:.4f},Kappa:{:.4f}, mIoU:{:.4f}, FWIoU: {:.4f},\nPrecision: {:.4f}, Recall: {:.4f}, f1_score: {:.4f}, f2_score: {:.4f}.".format(epoch+1,Acc,Kappa,mIoU,FWIoU,mPrecision,mRecall,mF1_score,mF2_score))
        print('Val Loss: %.3f' % (val_loss / num_batch)) 

        return val_loss/num_batch,Acc,Kappa,mIoU,mIoU0,mIoU1,mIoU2,FWIoU,mPrecision,Precision0,Precision1,Precision2,mRecall,Recall0,Recall1,Recall2,mF1_score,F1_score0,F1_score1,F1_score2,mF2_score,F2_score0,F2_score1,F2_score2

if __name__ == '__main__':
    main()


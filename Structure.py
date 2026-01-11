# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2026/01/11  23:47
# @function: 用于计算每轮预测准确率时，前向网络权值装载和预测分割结果的生成


import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

from PIL  import Image

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

class Model:
    def __init__(self, model_path, bands, num_class, model_type='unet', backbone='vggnet', atten_type='senet', img_size=256):
        self.model_path = model_path
        self.bands      = bands
        self.num_class  = num_class
        self.model_type = model_type
        self.backbone   = backbone
        self.img_size   = img_size
        self.atten_type = atten_type
        self.cuda       = torch.cuda.is_available()
        self.generate()

    def generate(self):
        if self.model_type == 'unet':
            self.model = Unet(bands=self.bands, num_classes=self.num_class, backbone=self.backbone, atten_type=self.atten_type)
        elif self.model_type == 'deeplab':
            self.model = DeepLab(bands=self.bands, num_classes=self.num_class, backbone=self.backbone, atten_type=self.atten_type)
        elif self.model_type == 'fcn':
            self.model = FCN16s(bands=self.bands, num_classes=self.num_class)
        elif self.model_type == 'hrnet':
            self.model = hrnet(bands=self.bands, num_classes=self.num_class, backbone=18, version='v2')                    # 18, 32, 48
        elif self.model_type == 'pspnet':
            self.model = PSPNet(bands=self.bands, num_classes=self.num_class, backbone=self.backbone, downsample_factor=8)
        elif self.model_type == 'refinenet':
            self.model = rf50(bands=self.bands, num_classes=self.num_class)
        elif self.model_type == 'enet':
            self.model = ENet(bands=self.bands, num_classes=self.num_class)
        elif self.model_type == 'segformer':
            self.model = Segformer(bands=self.bands, num_classes=self.num_class, backbone='b0')              # b0,b1,b2,b3,b4,b5
        elif self.model_type == 'setr':
            self.model = SETR(bands=self.bands, num_classes=self.num_class, backbone='Base', img_size=256) # Base,Large
        elif self.model_type == 'segnet':
            self.model = SegNet(bands=self.bands, num_classes=self.num_class)
        elif self.model_type == 'upernet':
            self.model = UperNet(bands=self.bands, num_classes=self.num_class) 
        elif self.model_type == 'segnext':
            self.model = SegNeXt(bands=self.bands, num_classes=self.num_class, backbone='T')    # T,S,B,L
        elif self.model_type == 'ocrnet':
            self.model = hrnetocr(bands=self.bands, num_classes=self.num_class, backbone=18) 
        elif self.model_type == 'mask2former':
            self.model = Mask2Former(bands=self.bands, num_classes=self.num_class)
        else:
            raise NotImplementedError('model type [%s] is not implemented, unet/deeplab/pspnet/hrnet/segnet/fcn/segformer/setr is supported!' %self.model_type)

        device   = torch.device('cuda:' if self.cuda else 'cpu')

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        try:
            state_dict = torch.load(self.model_path, map_location=device, weights_only=True)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[len("module."):] if key.startswith("module.") else key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict)
            self.model = self.model.eval()
            print(f"Model weights loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e


    def predict_small_patch(self, image):
        """Predict single image patch(256×256)"""

        assert image.ndim == 3, f"input image dimension show be [C, H, W], but get {image.shape} instead."
        c, h, w      = image.shape
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.cuda:
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            pr = self.model(image_tensor)         # [1, num_classes, H, W]
            pr = F.softmax(pr, dim=1)             # [1, num_classes, H, W]
            pr = pr.argmax(dim=1)                 # [1, H, W]
            pr = pr.squeeze(0).cpu().numpy()      # [H, W]
        return pr


    def predict_large_image(self, image, tile_size=256, overlap=1):
        """Predict large image"""

        assert image.ndim == 3, f"input image dimension show be [C, H, W], but get {image.shape} instead."
        c, h, w         = image.shape
        stride          = tile_size - overlap
        outputs         = np.zeros((h, w), dtype=np.float64)
        weights         = np.zeros((h, w), dtype=np.float32) 
        patches, coords = [], []
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                x_end   = min(i     + tile_size, h)
                y_end   = min(j     + tile_size, w)
                x_start = max(x_end - tile_size, 0)
                y_start = max(y_end - tile_size, 0)

                patches.append(image[:, x_start:x_end, y_start:y_end])
                coords.append((x_start, x_end, y_start, y_end))

        for patch, (x_start, x_end, y_start, y_end) in zip(patches, coords):
            prediction = self.predict_small_patch(patch)
            outputs[x_start:x_end, y_start:y_end] += prediction
            weights[x_start:x_end, y_start:y_end] += 1

        outputs = outputs / (weights + 1e-6)
        return np.round(outputs).astype(np.uint8)


    def get_small_predict_png(self, image):
        if image is None:
            raise ValueError("Image read error, please check path or format of file.")
        
        mask = self.predict_small_patch(image)
        palette = {
            0: [0,   0,   0],       # background
            1: [218, 218, 218],     # class 1(Clean Glacier)
            2: [0,  153, 0],       # class 2(Debris Covered Glacier)
        }
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in palette.items():
            rgb[mask == cls_id] = color
        return Image.fromarray(rgb)


    def get_large_predict_png(self, image):
        if image is None:
            raise ValueError("Image read error, please check path or format of file.")
        
        mask = self.predict_large_image(image)
        palette = {
            0: [0,   0,   0],       # background
            1: [218, 218, 218],     # class 1(Clean Glacier)
            2: [0,  153, 0],       # class 2(Debris Covered Glacier)
        }
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in palette.items():
            rgb[mask == cls_id] = color
        return Image.fromarray(rgb)

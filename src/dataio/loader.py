from PIL import Image
import numpy as np
from torchvision import transforms
from cvtorchvision import cvtransforms
from .rs_transforms_uint8 import RandomChannelDrop, RandomBrightness, RandomContrast, ToGray, GaussianBlur, Solarize



class TwoCropsTransform:
    """prende un'immagine e restituisce due versioni augmentate q e k, in base a base_transform"""
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]



def kaist_rgb_train_transforms(img_size=224):
    return cvtransforms.Compose([
        cvtransforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        cvtransforms.RandomApply([RandomBrightness(0.4), RandomContrast(0.4)], p=0.8),
        cvtransforms.RandomApply([ToGray(3)], p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.ToTensor(),
        cvtransforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])



def kaist_th_train_transforms(img_size=224):
    return cvtransforms.Compose([
        cvtransforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.ToTensor(),
        # cvtransforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), sono le statistiche di ImageNet per RGB (non thermal)
    ])



def build_kaist_transforms(img_size=224):
    t_rgb = TwoCropsTransform(kaist_rgb_train_transforms(img_size))
    t_th  = TwoCropsTransform(kaist_th_train_transforms(img_size))
    return t_rgb, t_th

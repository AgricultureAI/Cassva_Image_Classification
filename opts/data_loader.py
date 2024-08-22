import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from .fmix import sample_mask
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)


def get_img(path):
    '''使用 opencv 加载图片.
    由于历史原因，opencv 读取的图片格式是 bgr
    Args:
        path : str  图片文件路径 e.g '../data/train_img/1.jpg'
    '''
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def rand_bbox(size, lam):
    '''cutmix 的 bbox 截取函数
    Args:
        size : tuple 图片尺寸 e.g (256,256)
        lam  : float 截取比例
    Returns:
        bbox 的左上角和右下角坐标
        int,int,int,int
    '''
    W = size[0]  # 截取图片的宽度
    H = size[1]  # 截取图片的高度
    cut_rat = np.sqrt(1. - lam)  # 需要截取的 bbox 比例
    cut_w = np.int(W * cut_rat)  # 需要截取的 bbox 宽度
    cut_h = np.int(H * cut_rat)  # 需要截取的 bbox 高度

    cx = np.random.randint(W)  # 均匀分布采样，随机选择截取的 bbox 的中心点 x 坐标
    cy = np.random.randint(H)  # 均匀分布采样，随机选择截取的 bbox 的中心点 y 坐标

    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # 左上角 x 坐标
    bby1 = np.clip(cy - cut_h // 2, 0, H)  # 左上角 y 坐标
    bbx2 = np.clip(cx + cut_w // 2, 0, W)  # 右下角 x 坐标
    bby2 = np.clip(cy + cut_h // 2, 0, H)  # 右下角 y 坐标
    return bbx1, bby1, bbx2, bby2



class CassavaDataset(Dataset):
    '''木薯叶比赛数据加载类
    Attributes:
        __len__ : 数据的样本个数.
        __getitem__ : 索引函数.
    '''
    def __init__(self, df, data_root, transforms=None, output_label=True, one_hot_label=False, do_fmix=False,
            fmix_params={
                'alpha': 1.,
                'decay_power': 3.,
                'shape': (512, 512),
                'max_soft': 0.3,
                'reformulate': False
            },
            do_cutmix=False,
            cutmix_params={
                'alpha': 1,
            }):
        '''
        Args:
            df : DataFrame , 样本图片的文件名和标签
            data_root : str , 图片所在的文件路径，绝对路径
            transforms : object , 图片增强
            output_label : bool , 是否输出标签
            one_hot_label : bool , 是否进行 onehot 编码
            do_fmix : bool , 是否使用 fmix
            fmix_params :dict , fmix 的参数 {'alpha':1.,'decay_power':3.,'shape':(256,256),'max_soft':0.3,'reformulate':False}
            do_cutmix : bool, 是否使用 cutmix
            cutmix_params : dict , cutmix 的参数 {'alpha':1.}
        Raises:

        '''
        super().__init__()
        self.df = df.reset_index(drop=True).copy()  # 重新生成索引
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        if output_label:
            self.labels = self.df['label'].values
            if one_hot_label:
                self.labels = np.eye(self.df['label'].max() +
                                     1)[self.labels]  # 使用单位矩阵生成 onehot 编码

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        '''
        Args:
            index : int , 索引
        Returns:
            img, target(optional)
        '''
        if self.output_label:
            target = self.labels[index]

        img = get_img(
            os.path.join(self.data_root,
                         self.df.loc[index]['image_id']))  # 拼接地址，加载图片

        if self.transforms:  # 使用图片增强
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.uniform(
                0., 1., size=1)[0] > 0.5:  # 50% 概率触发 fmix 数据增强

            with torch.no_grad():
                lam, mask = sample_mask(
                    **self.fmix_params)  # 可以考虑魔改，使用 clip 规定上下限制

                fmix_ix = np.random.choice(self.df.index,
                                           size=1)[0]  # 随机选择待 mix 的图片
                fmix_img = get_img(
                    os.path.join(self.data_root,
                                 self.df.loc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)

                img = mask_torch * img + (1. - mask_torch) * fmix_img  # mix 图片

                rate = mask.sum() / float(img.size)  # 获取 mix 的 rate
                target = rate * target + (
                    1. - rate) * self.labels[fmix_ix]  # target 进行 mix

        if self.do_cutmix and np.random.uniform(
                0., 1., size=1)[0] > 0.5:  # 50% 概率触发 cutmix 数据增强
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img(
                    os.path.join(self.data_root,
                                 self.df.loc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(
                    np.random.beta(self.cutmix_params['alpha'],
                                   self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(cmix_img.shape[:2], lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2,
                                                        bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) *
                            (bby2 - bby1) / float(img.size))  # 获取 mix 的 rate
                target = rate * target + (
                    1. - rate) * self.labels[cmix_ix]  # target 进行 mix

        if self.output_label:
            return img, target
        else:
            return img



def get_train_transforms(img_size):
    return Compose([
        RandomResizedCrop(img_size, img_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)

# 验证集数据增强

def get_valid_transforms(img_size):
    return Compose([
        CenterCrop(img_size, img_size, p=1.),
        Resize(img_size, img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)




def prepare_dataloader(df, trn_idx, val_idx, data_root, trn_transform,
                       val_transform, bs, n_job):
    '''多进程数据生成器
    Args:
        df : DataFrame , 样本图片的文件名和标签
        trn_idx : ndarray , 训练集索引列表
        val_idx : ndarray , 验证集索引列表
        data_root : str , 图片文件所在路径
        trn_transform : object , 训练集图像增强器
        val_transform : object , 验证集图像增强器
        bs : int , 每次 batchsize 个数
        n_job : int , 使用进程数量
    Returns:
        train_loader, val_loader , 训练集和验证集的数据生成器
    '''
    train_ = df.loc[trn_idx, :].reset_index(drop=True)  # 重新生成索引
    valid_ = df.loc[val_idx, :].reset_index(drop=True)  # 重新生成索引

    train_ds = CassavaDataset(train_,
                              data_root,
                              transforms=trn_transform,
                              output_label=True,
                              one_hot_label=False,
                              do_fmix=False,
                              do_cutmix=False)
    valid_ds = CassavaDataset(valid_,
                              data_root,
                              transforms=val_transform,
                              output_label=True,
                              one_hot_label=False,
                              do_fmix=False,
                              do_cutmix=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=n_job,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=n_job,
    )

    return train_loader, val_loader
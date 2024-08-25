import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gradio as gr
import torch
import requests
import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from architecture import *
import argparse

from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)





parser = argparse.ArgumentParser(description="Cassva Image Classification Toolbox")
parser.add_argument('--data_name', type=str, default='SoybeanSeed', help='dataset name: Cassva, SoybeanSeed, ')
parser.add_argument('--method', type=str, default='SwinTransformer', help='method name：Efficientnet, Vit, SwinTransformer, AlexNet, MLPMixer, Timm_Efficientnet, Timm_Vit')
parser.add_argument('--pretrained_model_path', type=str, default=r'F:\agriculture\Soybean_Seeds_Classification\exp\SwinTransformer2024_08_25_08_23_37\models\model_epoch_SwinTransformer_fold_0_18.pth',  help='pretrained model directory')
parser.add_argument("--img_size", type=int, default=512, help='image size')
opt = parser.parse_args()

if opt.data_name.find('Cassva') >= 0:
    labels_e = ['Cassava Bacterial Blight (CBB)', 'Cassava Brown Streak Disease (CBSD)', 'Cassava Green Mottle (CGM)',
                'Cassava Mosaic Disease (CMD)', 'Healthy']
    labels_cn = ['木薯细菌性枯萎病', '木薯褐斑病', '木薯绿斑病', '木薯花叶病', '健康']
    opt.n_class = 5
elif opt.data_name.find('SoybeanSeed') >= 0:
    labels_e = ['Broken soybeans', 'Immature soybeans', 'Intact soybeans', 'Skin-damaged soybeans', 'Spotted soybeans']
    labels_cn = ['破碎的大豆', '未成熟大豆', '完整大豆', '皮肤受损的大豆', '大豆泥斑']
    opt.n_class = 5
    if opt.img_size >= 227:
        opt.img_size = 227

if opt.method.find('Vit') >= 0 or opt.method.find('Timm_Vit') >= 0 or opt.method.find('SwinTransformer') >= 0 or opt.method.find('MLPMixer') >= 0:
    opt.img_size = 224


model = model_generator(opt.method, opt.pretrained_model_path, n_class=opt.n_class).eval()

def get_valid_transforms(img_size):
    return Compose([
        CenterCrop(img_size, img_size, p=1.),
        Resize(img_size, img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

transform_test = get_valid_transforms(img_size=opt.img_size)

def predict(inp):
    inp = transform_test(image=inp)['image']
    inp = inp.unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels_cn[i]: float(prediction[i]) for i in range(5)}
    # max_index = prediction.argmax()
    # result = labels_cn[max_index]
    return confidences

demo = gr.Interface(fn=predict,
             inputs=[gr.Image(shape=(opt.img_size, opt.img_size),label='上传图像')],
             outputs=[gr.outputs.Label(num_top_classes=5, label='结果'), ],
             # examples=[["cheetah.jpg"]],
             )

demo.launch(server_port=7865)

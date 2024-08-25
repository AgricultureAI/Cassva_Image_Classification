import torch
import torch.nn as nn
# code from Timm
from .Timm_Efficientnet import TimmImgClassifier
from .Timm_Vit import ViTBase16
# my code
from .Aalexnet import AlexNet
from .Efficientnet import efficientnet_b4, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
    efficientnet_b5, efficientnet_b6, efficientnet_b7
from .Vit import vit_base_patch16_224, vit_base_patch16_224_in21k, vit_base_patch32_224, vit_base_patch32_224_in21k, \
    vit_large_patch16_224, vit_large_patch16_224_in21k, vit_large_patch32_224_in21k, vit_huge_patch14_224_in21k
from .SwinTransformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, \
    swin_base_patch4_window12_384, swin_base_patch4_window7_224_in22k, swin_base_patch4_window12_384_in22k, \
    swin_large_patch4_window7_224_in22k, swin_large_patch4_window12_384_in22k
from .MLPMixer import mixer_s32_224, mixer_s16_224, mixer_b32_224, mixer_b16_224, mixer_l32_224, mixer_l16_224, gmixer_12_224, gmixer_24_224, \
    resmlp_12_224, resmlp_24_224, resmlp_36_224, resmlp_big_24_224, gmlp_ti16_224, gmlp_s16_224, gmlp_b16_224


def model_generator(method, pretrained_model_path=None, pretrained=False, n_class=None):
    if method.find('Timm_Efficientnet') >= 0:
        model = TimmImgClassifier(model_arch='tf_efficientnet_b4_ns', n_class=n_class, pretrained=pretrained)
    elif method.find('Timm_Vit') >= 0:
        model = ViTBase16(n_classes=n_class, pretrained=pretrained)
    elif method.find('Efficientnet') >= 0:
        model = efficientnet_b4( num_classes=n_class)
    elif method.find('Vit') >= 0:
        model = vit_base_patch16_224(num_classes=n_class)
    elif method.find('SwinTransformer') >= 0:
        model = swin_tiny_patch4_window7_224(num_classes=n_class)
    elif method.find('AlexNet') >= 0:
        model = AlexNet(num_classes=n_class)
    elif method.find('MLPMixer') >= 0:
        model = mixer_b16_224(n_classes=n_class)
    else:
        print(f'Method {method} is not defined !!!!')

    if pretrained_model_path is None and pretrained is True and not method.find('Timm_Efficientnet') >= 0 and  not method.find('Timm_Vit')>= 0:
        try:
            if method.find('Efficientnet') >= 0:
                pretrained_model_path = 'ckpt_Image100/Efficientnet_b4.pth'
            elif method.find('Vit') >= 0:
                pretrained_model_path = 'ckpt_Image100/Vit_base_patch16_224.pth'
            elif method.find('SwinTransformer') >= 0:
                pretrained_model_path = 'ckpt_Image100/Swin_tiny_patch4_window7_224.pth'
            elif method.find('AlexNet') >= 0:
                pretrained_model_path = 'ckpt_Image100/Alexnet.pth'
            elif method.find('MLPMixer') >= 0:
                pretrained_model_path = 'ckpt_Image100/MLPMixer_b16_224.pth'
            else:
                print(f'Method {method} ckpt is not find !!!!')
        except:
            raise TypeError("No pretrained path")
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        try:
            checkpoint = torch.load(pretrained_model_path)
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                                  strict=True)
        except:
            if method.find('SwinTransformer') >= 0 or method.find('MLPMixer') >= 0 or method.find('Vit') >= 0:
                weights_dict = torch.load(pretrained_model_path)
                weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
                for k in list(weights_dict.keys()):
                    if "head" in k:
                        del weights_dict[k]
                model.load_state_dict(weights_dict, strict=False)

            elif method.find('AlexNet') >= 0 or method.find('Efficientnet') >= 0:
                weights_dict = torch.load(pretrained_model_path)
                weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
                for k in list(weights_dict.keys()):
                    if "classifier" in k:
                        del weights_dict[k]
                model.load_state_dict(weights_dict, strict=False)

            elif method.find('MambaVision') >= 0:
                weights_dict = torch.load(pretrained_model_path)
                weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
                model.load_state_dict(weights_dict, strict=False)
                in_channel = model.head.in_features
                model.head = nn.Linear(in_channel, n_class)

    return model

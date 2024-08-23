import torch
import torch.nn as nn
from .Aalexnet import AlexNet
from .Efficientnet import efficientnet_b4
from .Vit import vit_base_patch16_224
from .SwinTransformer import swin_tiny_patch4_window7_224
from .MLPMixer import mixer_b16_224

def model_generator(method, pretrained_model_path=None, pretrained=False, n_class=None):
    if method.find('Efficientnet') >= 0:
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
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        try:
            checkpoint = torch.load(pretrained_model_path)
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                                  strict=True)
        except:
            if method.find('SwinTransformer') >= 0 or method.find('ConvNeXt') >= 0 or method.find('RegNet') >= 0 or method.find('VIM') >= 0 or method.find('MLPMixer') >= 0 or method.find('TransNeXt') >= 0:
                weights_dict = torch.load(pretrained_model_path)
                weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
                for k in list(weights_dict.keys()):
                    if "head" in k:
                        del weights_dict[k]
                model.load_state_dict(weights_dict, strict=False)

            elif method.find('Mobile_Vit') >= 0 or method.find('Densenet') >= 0 or method.find('MobileNetV2') >= 0 or method.find('AlexNet') >= 0 or method.find('vgg') >= 0 or method.find('Mobilenet_v3') >= 0:
                weights_dict = torch.load(pretrained_model_path)
                weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
                for k in list(weights_dict.keys()):
                    if "classifier" in k:
                        del weights_dict[k]
                model.load_state_dict(weights_dict, strict=False)

            elif method.find('GoogLeNet') >= 0 or method.find('resnet101') >= 0 or method.find('shufflenet') >= 0:
                weights_dict = torch.load(pretrained_model_path)
                weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
                for k in list(weights_dict.keys()):
                    if "fc" in k:
                        del weights_dict[k]
                model.load_state_dict(weights_dict, strict=False)
            elif method.find('MambaVision') >= 0:
                weights_dict = torch.load(pretrained_model_path)
                weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
                model.load_state_dict(weights_dict, strict=False)
                in_channel = model.head.in_features  # 获取最后一层线性层的in_channel
                model.head = nn.Linear(in_channel, n_class)

    return model

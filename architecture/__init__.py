import torch
from .Efficientnet import TimmImgClassifier
from .Vit import ViTBase16

def model_generator(method, pretrained_model_path=None, pretrained=False):
    if method.find('Efficientnet') >= 0:
        model = TimmImgClassifier(model_arch='tf_efficientnet_b4_ns', n_class=5, pretrained=pretrained)
    elif method.find('Vit') >= 0:
        model = ViTBase16(n_classes=5, pretrained=pretrained)

    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    return model

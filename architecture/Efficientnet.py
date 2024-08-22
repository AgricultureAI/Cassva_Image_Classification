from torch import nn
import timm

class TimmImgClassifier(nn.Module):
    def __init__(self, model_arch='tf_efficientnet_b4_ns', n_class=5, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
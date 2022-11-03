import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0, swin_s, convnext_small

__all__ = ["BaseModule"]


class BaseModule(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)
        return

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint)
        return
    
    def data_parallel(self, device_ids):
        return nn.DataParallel(self, device_ids=device_ids)


class EfficientNet_B0(BaseModule):
    def __init__(self, num_classes):
        super(EfficientNet_B0, self).__init__()
        self.backbone = efficientnet_b0(weights='IMAGENET1K_V1')
        self.fc = nn.Linear(self.backbone.classifier[1].out_features, num_classes)

    def forward(self, inputs):
        return self.fc(self.backbone(inputs))


class RegNet_Y_16(BaseModule):
    def __init__(self, num_classes):
        super(RegNet_Y_16, self).__init__()
        self.backbone = regnet_y_16gf(weights='IMAGENET1K_V1')
        self.fc = nn.Linear(self.backbone.fc.out_features, num_classes)

    def forward(self, inputs):
        return self.fc(self.backbone(inputs))


class Swin_S(BaseModule):
    def __init__(self, num_classes):
        super(Swin_S, self).__init__()
        self.backbone = swin_s(weights='IMAGENET1K_V1')
        self.fc = nn.Linear(self.backbone.head.out_features, num_classes)

    def forward(self, inputs):
        return self.fc(self.backbone(inputs))


class ConvNext_S(BaseModule):
    def __init__(self, num_classes):
        super(ConvNext_S, self).__init__()
        self.backbone = convnext_small(weights='IMAGENET1K_V1')
        self.fc = nn.Linear(self.backbone.classifier[-1].out_features, num_classes)

    def forward(self, inputs):
        return self.fc(self.backbone(inputs))

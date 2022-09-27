import torch
import torch.nn as nn
from torchvision.models import convnext_base
from .base import BaseModule


class ConvNeXt_B(BaseModule):
    def __init__(self, num_classes=219):
        super(ConvNeXt_B, self).__init__()
        self.model = convnext_base(pretrained=True)
        self.model.classifier[2] = torch.nn.Linear(
            self.model.classifier[2].in_features, num_classes)


    def forward(self, inputs, *args, **kwargs):
        outputs = self.model(inputs)

        return outputs

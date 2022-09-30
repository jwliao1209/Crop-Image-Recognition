import torch 
import torch.nn as nn
from torchvision.models import densenet121
from .base import BaseModule

class DenseNet121(BaseModule):
    def __init__(self, num_classes=33):
        super(DenseNet121, self).__init__()
        self.model = densenet121(pretrained='IMAGENET1K_V1')
        self.linear  = nn.Linear(1000, num_classes)

    def forward(self, inputs, use_mc_loss=False, *args, **kwargs):
        x = self.model(inputs)
        outputs = self.linear(x)
        return outputs
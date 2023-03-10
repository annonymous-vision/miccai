import pickle

import timm
import torch
from torch import nn


class BaseLine_MIMIC(nn.Module):
    def __init__(self, args):
        super(BaseLine_MIMIC, self).__init__()
        self.args = args
        self.feature_store = {}
        # define feature extractors
        self.backbone = timm.create_model(args.arch, pretrained=self.args.pretrained, features_only=True)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=1024 * 16 * 16, out_features=107, bias=True)

    def forward(self, x):
        phi = self.backbone(x)[-1]
        logits = self.linear(self.flatten(phi))
        return logits

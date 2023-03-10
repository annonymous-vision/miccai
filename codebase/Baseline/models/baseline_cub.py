import numpy as np
import torch

import utils
from BB.models.VIT import CONFIGS, VisionTransformer
from BB.models.t import Logistic_Regression_t


class BaseLine_CUB(torch.nn.Module):
    def __init__(self, args, test=False):
        super(BaseLine_CUB, self).__init__()
        self.args = args
        self.backbone = self.get_model()
        input_size_t = self.get_input_size_t()
        self.t = Logistic_Regression_t(
            ip_size=input_size_t, op_size=len(args.concept_names), flattening_type=args.flattening_type
        )
        self.test = test

    def forward(self, x):
        phi = self.get_phi_x(x)
        logits_concepts = self.t(phi)
        if self.test:
            return logits_concepts, phi
        else:
            return logits_concepts

    def get_phi_x(self, x):
        if self.args.arch == "ResNet50" or self.args.arch == "ResNet101" or self.args.arch == "ResNet152":
            _ = self.backbone(x)
            # feature_x = get_flattened_x(bb.feature_store[layer], flattening_type)
            feature_x = self.backbone.feature_store[self.args.layer]
            return feature_x
        elif self.args.arch == "ViT-B_16":
            logits, tokens = self.backbone(x)
            return tokens[:, 0]

    def get_model(self):
        if self.args.arch == "ResNet50" or self.args.arch == "ResNet101":
            return utils.get_model(
                self.args.arch, self.args.dataset, self.args.pretrained, len(self.args.labels), layer=self.args.layer
            )
        elif self.args.arch == "ViT-B_16":
            _config = CONFIGS[self.args.arch]
            _config.split = "non-overlap"
            _config.slide_step = 12
            _img_size = self.args.img_size
            _smoothing_value = 0.0
            _num_classes = len(self.args.labels)

            model = VisionTransformer(
                _config, _img_size, zero_head=True, num_classes=_num_classes, smoothing_value=_smoothing_value
            )

            pre_trained = "Path/PhD/miccai/checkpoints/pretrained_VIT/ViT-B_16.npz"
            checkpoint = np.load(pre_trained)
            model.load_from(checkpoint)
            return model

    def get_input_size_t(self):
        if self.args.arch == "ResNet50" or self.args.arch == "ResNet101" or self.args.arch == "ResNet152":
            return self.backbone.base_model.fc.weight.shape[1] \
                if self.args.layer == "layer4" else int(self.backbone.base_model.fc.weight.shape[1] / 2)
        elif self.args.arch == "ViT-B_16":
            return 768

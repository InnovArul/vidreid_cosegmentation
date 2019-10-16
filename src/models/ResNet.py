from __future__ import absolute_import

import torch
import os
import sys
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import utils
from .aggregation_layers import AGGREGATION
from .cosam import COSEG_ATTENTION


def get_ResNet(net_type):
    if net_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2])
    elif net_type == "senet101":
        model = models.resnet101(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2])
    else:
        assert False, "unknown ResNet type : " + net_type

    return model


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("instantiating " + self.__class__.__name__)

    def forward(self, x, b, t):
        return [x, None, None]


class CosegAttention(nn.Module):
    def __init__(self, attention_types, num_feat_maps, h_w, t):
        super().__init__()
        print("instantiating " + self.__class__.__name__)
        self.attention_modules = nn.ModuleList()

        for i, attention_type in enumerate(attention_types):
            if attention_type in COSEG_ATTENTION:
                self.attention_modules.append(
                    COSEG_ATTENTION[attention_type](
                        num_feat_maps[i], h_w=h_w[i], t=t)
                )
            else:
                assert False, "unknown attention type " + attention_type

    def forward(self, x, i, b, t):
        return self.attention_modules[i](x, b, t)


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes,
        net_type="resnet50",
        attention_types=["NONE", "NONE", "NONE", "NONE"],
        aggregation_type="tp",
        seq_len=4,
        **kwargs
    ):
        super(ResNet, self).__init__()
        print(
            "instantiating "
            + self.__class__.__name__
            + " net type"
            + net_type
            + " from "
            + __file__
        )
        print("attention type", attention_types)

        # base network instantiation
        self.base = get_ResNet(net_type=net_type)
        self.feat_dim = 2048

        # attention modules
        self.num_feat_maps = [256, 512, 1024, 2048]
        self.h_w = [(64, 32), (32, 16), (16, 8), (8, 4)]
        if aggregation_type == "ta":
            self.h_w = [(64, 32), (32, 16), (16, 8), (8, 4)]
        else:
            utils.set_stride(self.base[7], 1)
            self.h_w = [(64, 32), (32, 16), (16, 8), (16, 8)]

        print(self.h_w)
        self.attention_modules = CosegAttention(
            attention_types, num_feat_maps=self.num_feat_maps, h_w=self.h_w, t=seq_len
        )

        # aggregation module
        self.aggregation = AGGREGATION[aggregation_type](
            self.feat_dim, h_w=self.h_w[-1], t=seq_len
        )

        # classifier
        self.classifier = nn.Linear(self.aggregation.feat_dim, num_classes)

    def base_layer0(self, x):
        x = self.base[0](x)
        x = self.base[1](x)
        x = self.base[2](x)
        x = self.base[3](x)
        x = self.base[4](x)
        return x

    def base_layer1(self, x):
        return self.base[5](x)

    def base_layer2(self, x):
        return self.base[6](x)

    def base_layer3(self, x):
        return self.base[7](x)

    def extract_features(self, x, b, t):
        features_in = x
        attentions = []

        for index in range(len(self.num_feat_maps)):
            features_before_attention = getattr(self, "base_layer" + str(index))(
                features_in
            )
            features_out, channelwise, spatialwise = self.attention_modules(
                features_before_attention, index, b, t
            )
            features_in = features_out

            # note down the attentions
            attentions.append(
                (features_before_attention, features_out, channelwise, spatialwise)
            )

        return features_out, attentions

    def return_values(
        self, features, logits, attentions, is_training, return_attentions
    ):
        buffer = []
        if not is_training:
            buffer.append(features)
            if not return_attentions:
                return features
        else:
            buffer.append(logits)
            buffer.append(features)

        if return_attentions:
            buffer.append(attentions)

        return buffer

    def forward(self, x, return_attentions=False):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))

        final_spatial_features, attentions = self.extract_features(x, b, t)
        f = self.aggregation(final_spatial_features, b, t)

        if not self.training:
            return self.return_values(
                f, None, attentions, self.training, return_attentions
            )

        y = self.classifier(f)
        return self.return_values(f, y, attentions, self.training, return_attentions)


# all mutual correlation attention models
def ResNet50_COSAM45_TP(num_classes, **kwargs):
    return ResNet(
        num_classes,
        net_type="resnet50",
        aggregation_type="tp",
        attention_types=["NONE", "NONE",
                         "COSAM", "COSAM"],
        **kwargs
    )


def ResNet50_COSAM45_TA(num_classes, **kwargs):
    return ResNet(
        num_classes,
        net_type="resnet50",
        aggregation_type="ta",
        attention_types=["NONE", "NONE",
                         "COSAM", "COSAM"],
        **kwargs
    )


def ResNet50_COSAM45_RNN(num_classes, **kwargs):
    return ResNet(
        num_classes,
        net_type="resnet50",
        aggregation_type="rnn",
        attention_types=["NONE", "NONE",
                         "COSAM", "COSAM"],
        **kwargs
    )


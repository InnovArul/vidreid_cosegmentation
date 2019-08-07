from __future__ import absolute_import

import torch
import os
import sys
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
import utils as utils

this_path = os.path.split(__file__)[0]
sys.path.append(this_path)
from .aggregation_layers import AGGREGATION
from .cosam import COSEG_ATTENTION
import senet

def get_SENet(net_type):
    if net_type == "senet50":
        model = senet.se_resnet50(pretrained=True)
    elif net_type == "senet101":
        model = senet.se_resnet101(pretrained=True)
    else:
        assert False, "unknown SE ResNet type : " + net_type

    return model


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


class SE_ResNet(nn.Module):
    def __init__(
        self,
        num_classes,
        net_type="senet50",
        attention_types=["NONE", "NONE", "NONE", "NONE", "NONE"],
        aggregation_type="tp",
        seq_len=4,
        is_baseline=False,
        **kwargs
    ):
        super(SE_ResNet, self).__init__()
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
        self.base = get_SENet(net_type=net_type)
        self.feat_dim = self.base.feature_dim

        # attention modules
        self.num_feat_maps = [64, 256, 512, 1024, 2048]
        self.h_w = [(64, 32), (64, 32), (32, 16), (16, 8), (8, 4)]

        # allow reducing spatial dimension for Temporal attention (ta) to keep the params at a manageable number
        # according to the baseline paper
        if aggregation_type == "ta":
            self.h_w = [(64, 32), (64, 32), (32, 16), (16, 8), (8, 4)]
        else:
            utils.set_stride(self.base.layer4, 1)
            self.h_w = [(64, 32), (64, 32), (32, 16), (16, 8), (16, 8)]

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

    def extract_features(self, x, b, t):
        features_in = x
        attentions = []

        for index in range(5):
            features_before_attention = getattr(self.base, "layer" + str(index))(
                features_in
            )
            features_out, channelwise, spatialwise = self.attention_modules(
                features_before_attention, index, b, t
            )
            # print(features_out.shape)
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


# all seresnet50 models
def SE_ResNet50_TP(num_classes, **kwargs):
    return SE_ResNet(
        num_classes,
        net_type="senet50",
        aggregation_type="tp",
        is_baseline=True,
        **kwargs
    )


def SE_ResNet50_TA(num_classes, **kwargs):
    return SE_ResNet(
        num_classes,
        net_type="senet50",
        aggregation_type="ta",
        is_baseline=True,
        **kwargs
    )


def SE_ResNet50_RNN(num_classes, **kwargs):
    return SE_ResNet(
        num_classes,
        net_type="senet50",
        aggregation_type="rnn",
        is_baseline=True,
        **kwargs
    )


# all mutual correlation attention models
def SE_ResNet50_COSAM45_TP(num_classes, **kwargs):
    return SE_ResNet(
        num_classes,
        net_type="senet50",
        aggregation_type="tp",
        attention_types=[
            "NONE",
            "NONE",
            "NONE",
            "COSAM",
            "COSAM",
        ],
        **kwargs
    )


def SE_ResNet50_COSAM45_TA(num_classes, **kwargs):
    return SE_ResNet(
        num_classes,
        net_type="senet50",
        aggregation_type="ta",
        attention_types=[
            "NONE",
            "NONE",
            "NONE",
            "COSAM",
            "COSAM",
        ],
        **kwargs
    )


def SE_ResNet50_COSAM45_RNN(num_classes, **kwargs):
    return SE_ResNet(
        num_classes,
        net_type="senet50",
        aggregation_type="rnn",
        attention_types=[
            "NONE",
            "NONE",
            "NONE",
            "COSAM",
            "COSAM",
        ],
        **kwargs
    )

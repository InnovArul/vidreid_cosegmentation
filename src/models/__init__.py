from __future__ import absolute_import

from .ResNet import *
from .SE_ResNet import *

__factory = {
    # ResNet50 network
    "resnet50_cosam45_tp": ResNet50_COSAM45_TP,
    "resnet50_cosam45_ta": ResNet50_COSAM45_TA,
    "resnet50_cosam45_rnn": ResNet50_COSAM45_RNN,

    # Squeeze and Expand network
    "se_resnet50_cosam45_tp": SE_ResNet50_COSAM45_TP,
    "se_resnet50_cosam45_ta": SE_ResNet50_COSAM45_TA,
    "se_resnet50_cosam45_rnn": SE_ResNet50_COSAM45_RNN,

}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)

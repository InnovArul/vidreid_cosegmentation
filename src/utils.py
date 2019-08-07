from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json, math
import os.path as osp
import more_itertools as mit
import torch, torch.nn as nn
import numpy as np
import inspect

is_print_once_enabled = True


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def disable_all_print_once():
    global is_print_once_enabled
    is_print_once_enabled = False


@static_vars(lines={})
def print_once(msg):
    # return from the function if the API is disabled
    global is_print_once_enabled
    if not is_print_once_enabled:
        return

    from inspect import getframeinfo, stack

    caller = getframeinfo(stack()[1][0])
    current_file_line = "%s:%d" % (caller.filename, caller.lineno)

    # if the current called file and line is not in buffer print once
    if current_file_line not in print_once.lines:
        print(msg)
        print_once.lines[current_file_line] = True


def get_executing_filepath():
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    return os.path.split(filename)[0]


def set_stride(module, stride):
    """
    
    """
    print("setting stride of ", module, " to ", stride)
    for internal_module in module.modules():
        if isinstance(internal_module, nn.Conv2d) or isinstance(
            internal_module, nn.MaxPool2d
        ):
            internal_module.stride = stride

    return internal_module


def get_gaussian_kernel(channels, kernel_size=5, mean=0, sigma=[1, 4]):
    # CONVERT INTO NP ARRAY
    sigma_ = torch.zeros((2, 2)).float()
    sigma_[0, 0] = sigma[0]
    sigma_[1, 1] = sigma[1]
    sigma = sigma_

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.linspace(-1, 1, kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    variance = (sigma @ sigma.t()).float()
    inv_variance = torch.inverse(variance)

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * torch.det(variance))) * torch.exp(
        -torch.sum(
            ((xy_grid - mean) @ inv_variance.unsqueeze(0)) * (xy_grid - mean), dim=-1
        )
        / 2
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(1, channels, 1, 1)
    return gaussian_kernel


def mkdir_if_missing(directory):
    """to create a directory
    
    Arguments:
        directory {str} -- directory path
    """

    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# function to freeze certain module's weight
def freeze_weights(to_be_freezed):
    for param in to_be_freezed.parameters():
        param.requires_grad = False

    for module in to_be_freezed.children():
        for param in module.parameters():
            param.requires_grad = False


def load_pretrained_model(model, pretrained_model_path, verbose=False):
    """To load the pretrained model considering the number of keys and their sizes
    
    Arguments:
        model {loaded model} -- already loaded model
        pretrained_model_path {str} -- path to the pretrained model file
    
    Raises:
        IOError -- if the file path is not found
    
    Returns:
        model -- model with loaded params
    """

    if not os.path.exists(pretrained_model_path):
        raise IOError("Can't find pretrained model: {}".format(pretrained_model_path))

    print("Loading checkpoint from '{}'".format(pretrained_model_path))
    pretrained_state = torch.load(pretrained_model_path)["state_dict"]
    print(len(pretrained_state), " keys in pretrained model")

    current_model_state = model.state_dict()
    print(len(current_model_state), " keys in current model")
    pretrained_state = {
        key: val
        for key, val in pretrained_state.items()
        if key in current_model_state and val.size() == current_model_state[key].size()
    }

    print(
        len(pretrained_state),
        " keys in pretrained model are available in current model",
    )
    current_model_state.update(pretrained_state)
    model.load_state_dict(current_model_state)

    if verbose:
        non_available_keys_in_pretrained = [
            key
            for key, val in pretrained_state.items()
            if key not in current_model_state
            or val.size() != current_model_state[key].size()
        ]
        non_available_keys_in_current = [
            key
            for key, val in current_model_state.items()
            if key not in pretrained_state or val.size() != pretrained_state[key].size()
        ]

        print(
            "not available keys in pretrained model: ", non_available_keys_in_pretrained
        )
        print("not available keys in current model: ", non_available_keys_in_current)

    return model


def get_currenttime_prefix():
    """to get a prefix of current time
    
    Returns:
        [str] -- current time encoded into string
    """

    from time import localtime, strftime

    return strftime("%d-%b-%Y_%H:%M:%S", localtime())


def get_learnable_params(model):
    """to get the list of learnable params
    
    Arguments:
        model {model} -- loaded model
    
    Returns:
        list -- learnable params
    """

    # list down the names of learnable params
    details = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            details.append((name, param.shape))
    print("learnable params (" + str(len(details)) + ") : ", details)

    #  short list the params which has requires_grad as true
    learnable_params = [param for param in model.parameters() if param.requires_grad]

    print(
        "Model size: {:.5f}M".format(
            sum(p.numel() for p in learnable_params) / 1000000.0
        )
    )
    return learnable_params


def get_features(model, imgs, test_num_tracks):
    """to handle higher seq length videos due to OOM error
    specifically used during test
    
    Arguments:
        model -- model under test
        imgs -- imgs to get features for
    
    Returns:
        features 
    """

    # handle chunked data
    all_features = []

    for test_imgs in mit.chunked(imgs, test_num_tracks):
        current_test_imgs = torch.stack(test_imgs)
        num_current_test_imgs = current_test_imgs.shape[0]
        # print(current_test_imgs.shape)
        features = model(current_test_imgs)
        features = features.view(num_current_test_imgs, -1)
        all_features.append(features)

    return torch.cat(all_features)


def get_spatial_features(model, imgs, test_num_tracks):
    """to handle higher seq length videos due to OOM error
    specifically used during test
    
    Arguments:
        model -- model under test
        imgs -- imgs to get features for
    
    Returns:
        features 
    """

    # handle chunked data
    all_features, all_spatial_features = [], []

    for test_imgs in mit.chunked(imgs, test_num_tracks):
        current_test_imgs = torch.stack(test_imgs)
        num_current_test_imgs = current_test_imgs.shape[0]
        features, spatial_feats = model(current_test_imgs)
        features = features.view(num_current_test_imgs, -1)

        all_spatial_features.append(spatial_feats)
        all_features.append(features)

    return torch.cat(all_features), torch.cat(all_spatial_features)


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath="checkpoint.pth.tar"):
    mkdir_if_missing(osp.dirname(fpath))
    print("saving model to " + fpath)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), "best_model.pth.tar"))


def open_all_layers(model):
    """
    Open all layers in model for training.

    Args:
    - model (nn.Module): neural net model.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    """
    Open specified layers in model for training while keeping
    other layers frozen.
    
    Args:
    - model (nn.Module): neural net model.
    - open_layers (list): list of layer names.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )

    # check if all the open layers are there in model
    all_names = [name for name, module in model.named_children()]
    for tobeopen_layer in open_layers:
        assert tobeopen_layer in all_names, "{} not in model".format(tobeopen_layer)

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, "a")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def read_json(fpath):
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


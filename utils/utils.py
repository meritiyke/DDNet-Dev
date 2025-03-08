import random
import numpy as np
import torch
import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    """
    Initialize weights for convolutional and normalization layers.
    """
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)  # Initialize convolutional layer weights
        elif isinstance(m, norm_layer):
            m.eps = bn_eps  # Set epsilon for batch normalization
            m.momentum = bn_momentum  # Set momentum for batch normalization
            nn.init.constant_(m.weight, 1)  # Initialize batch norm weight to 1
            nn.init.constant_(m.bias, 0)  # Initialize batch norm bias to 0


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    """
    Initialize weights for a list of modules or a single module.
    """
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)


def setup_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CUDA operations


def netParams(model):
    """
    Compute the total number of parameters in the model.
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters
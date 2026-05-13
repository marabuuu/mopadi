"""
Various utilities for neural networks.
"""

from contextlib import nullcontext
from enum import Enum
import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.utils.checkpoint

import torch.nn.functional as F


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    # @th.jit.script
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(-math.log(max_period) *
                   th.arange(start=0, end=half, dtype=th.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat(
            [embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    # torch's gradient checkpoint works with automatic mixed precision, given torch >= 1.8
    if flag:
        autocast_enabled = th.is_autocast_enabled()
        autocast_dtype = th.get_autocast_gpu_dtype() if autocast_enabled else None

        def context_fn():
            if autocast_enabled:
                return (
                    th.autocast(device_type="cuda", dtype=autocast_dtype),
                    th.autocast(device_type="cuda", dtype=autocast_dtype),
                )
            return nullcontext(), nullcontext()

        # use_reentrant=False: non-reentrant checkpoint does not re-run the
        # forward inside the backward pass.  Reentrant mode (the default) fires
        # DDP's autograd hooks twice per parameter (once per reentrant backward
        # re-entry), causing "variable marked ready twice" crashes under DDP.
        return torch.utils.checkpoint.checkpoint(
            func, *args, preserve_rng_state=preserve_rng_state,
            use_reentrant=False, context_fn=context_fn)
    else:
        return func(*args)

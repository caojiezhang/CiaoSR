# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import load_checkpoint

# from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger

from torch import Tensor
import pdb


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class Cos(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.cos(input)

class Sin(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)

class MLPRefiner(nn.Module):
    """Multilayer perceptrons (MLPs), refiner used in LIIF.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_list (list[int]): List of hidden dimensions.
    """

    def __init__(self, in_dim, out_dim, hidden_list=None, act=None):
        super().__init__()
        layers = []
        lastv = in_dim
        if hidden_list:
            for hidden in hidden_list:
                layers.append(nn.Linear(lastv, hidden))
                if act == 'cos':
                    layers.append(Cos())
                elif act == 'sin':
                    layers.append(Sin())
                else:
                    layers.append(nn.ReLU())
                lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): The input of MLP.

        Returns:
            Tensor: The output of MLP.
        """
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class PosMLPRefiner(nn.Module):
    """Multilayer perceptrons (MLPs), refiner used in LIIF.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_list (list[int]): List of hidden dimensions.
    """

    def __init__(self, in_dim, out_dim, hidden_list=None, is_pos=True):
        super().__init__()
        self.layers = []
        lastv = in_dim
        if hidden_list:
            for hidden in hidden_list:
                self.layers.append(nn.Linear(lastv, hidden))
                self.layers.append(nn.ReLU())
                if is_pos:
                    self.layers.append(PositionalEncoding1D(hidden))
                lastv = hidden
        self.layers.append(nn.Linear(lastv, out_dim))
        # self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): The input of MLP.

        Returns:
            Tensor: The output of MLP.
        """
        B, C, D = x.shape
        shape = x.shape[:-1]

        for layer in self.layers:
            if 'Pos' in str(layer):
                x += layer(x)
            else:
                x = layer(x.view(B*C, -1))
                x = x.view(*shape, -1)
        return x

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


if __name__ == '__main__':
    m = PosMLPRefiner(64, 64, [32, 32], True)
    x = torch.rand(3, 32, 64)
    print(m(x).shape)

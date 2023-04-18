# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmcv import imresize

from mmedited.utils.utils_image import im_resize


class RandomBicubicSampling:
    """Generate LQ image from GT (and crop), which will randomly pick a scale.

    Args:
        scale_min (float): The minimum of upsampling scale, inclusive.
            Default: 1.0.
        scale_max (float): The maximum of upsampling scale, exclusive.
            Default: 4.0.
        patch_size (int): The cropped lr patch size.
            Default: None, means no crop.

        Scale will be picked in the range of [scale_min, scale_max).
    """

    def __init__(self,
                 scale_min=1.0,
                 scale_max=4.0,
                 patch_size=None,
                 ):
        assert scale_max >= scale_min
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.patch_size = patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation. 'gt' is required.

        Returns:
            dict: A dict containing the processed data and information.
                modified 'gt', supplement 'lq' and 'scale' to keys.
        """
        img = results['gt']
        scale = np.random.uniform(self.scale_min, self.scale_max)

        if self.patch_size is None:
            h_lr = math.floor(img.shape[-3] / scale + 1e-9)
            w_lr = math.floor(img.shape[-2] / scale + 1e-9)
            img = img[:round(h_lr * scale), :round(w_lr * scale), :]
            img_down = resize_fn(img, (w_lr, h_lr), scale)
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.patch_size
            w_hr = round(w_lr * scale)
            x0 = np.random.randint(0, img.shape[-3] - w_hr)
            y0 = np.random.randint(0, img.shape[-2] - w_hr)
            crop_hr = img[x0:x0 + w_hr, y0:y0 + w_hr, :]
            crop_lr = resize_fn(crop_hr, w_lr, scale)
        results['gt'] = crop_hr
        results['lq'] = crop_lr
        results['scale'] = scale

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f' scale_min={self.scale_min}, '
                     f'scale_max={self.scale_max}, '
                     f'patch_size={self.patch_size}, '
                     )

        return repr_str


def resize_fn(img, size, scale):
    """Resize the given image to a given size.

    Args:
        img (ndarray | torch.Tensor): The input image.
        size (int | tuple[int]): Target size w or (w, h).
        interpolation (str): Interpolation method, only "bicubic".

    Returns:
        ndarray | torch.Tensor: `resized_img`, whose type is same as `img`.
    """
    if isinstance(size, int):
        size = (size, size)
    assert img.min() >=0 and img.max() <= 1, "img shound in [0, 1]"

    if isinstance(img, np.ndarray) or isinstance(img, torch.Tensor):
        return im_resize(img, size, 1/scale)
    else:
        raise TypeError('img should got np.ndarray or torch.Tensor,'
                        f'but got {type(img)}')


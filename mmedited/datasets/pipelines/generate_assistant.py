# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmedit.datasets.pipelines.utils import make_coord


class GenerateCoordinateAndCell1:
    """Generate coordinate and cell.
    Generate coordinate from the desired size of SR image.
        Train or val:
            1. Generate coordinate from GT.
            2. Reshape GT image to (HgWg, 3) and transpose to (3, HgWg).
                where `Hg` and `Wg` represent the height and width of GT.
        Test:
            Generate coordinate from LQ and scale or target_size.
    Then generate cell from coordinate.
    Args:
        sample_quantity (int): The quantity of samples in coordinates.
            To ensure that the GT tensors in a batch have the same dimensions.
            Default: None.
        scale (float): Scale of upsampling.
            Default: None.
        target_size (tuple[int]): Size of target image.
            Default: None.
    The priority of getting 'size of target image' is:
        1, results['gt'].shape[-2:]
        2, results['lq'].shape[-2:] * scale
        3, target_size
    """

    def __init__(self, sample_quantity=None, scale=None, target_size=None, is_shuffle=True):
        self.sample_quantity = sample_quantity
        self.scale = scale
        self.target_size = target_size
        self.is_shuffle = is_shuffle

    def __call__(self, results):
        """Call function.
        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.
                Require either in results:
                    1. 'lq' (tensor), whose shape is similar as (3, H, W).
                    2. 'gt' (tensor), whose shape is similar as (3, H, W).
                    3. None, the premise is
                        self.target_size and len(self.target_size) >= 2.
        Returns:
            dict: A dict containing the processed data and information.
                Reshape 'gt' to (-1, 3) and transpose to (3, -1) if 'gt'
                in results.
                Add 'coord' and 'cell'.
        """
        # generate hr_coord (and hr_rgb)
        if 'gt' in results:
            crop_hr = results['gt']
            self.target_size = crop_hr.shape
            hr_rgb = crop_hr.contiguous().view(3, -1).permute(1, 0)
            results['gt'] = hr_rgb
            if 'gt_unsharp' in results:
                crop_hr1 = results['gt_unsharp']
                results['gt_unsharp'] = crop_hr1.contiguous().view(3, -1).permute(1, 0)
        elif self.scale is not None and 'lq' in results:
            _, h_lr, w_lr = results['lq'].shape
            self.target_size = (round(h_lr * self.scale),
                                round(w_lr * self.scale))
        else:
            assert self.target_size is not None
            assert len(self.target_size) >= 2
        hr_coord = make_coord(self.target_size[-2:])

        if self.sample_quantity is not None and 'gt' in results and 'gt_unsharp' in results:
            if self.is_shuffle:
                sample_lst = np.random.choice(
                    len(hr_coord), self.sample_quantity, replace=False)
            else:
                if len(hr_coord) == self.sample_quantity:
                    idx_start = 0
                else:
                    idx_start = int(np.random.choice(len(hr_coord)-self.sample_quantity, 1, replace=False))
                sample_lst = list(range(idx_start, idx_start+self.sample_quantity, 1))

            hr_coord = hr_coord[sample_lst]
            results['gt'] = results['gt'][sample_lst]
            results['gt_unsharp'] = results['gt_unsharp'][sample_lst]

        # Preparations for cell decoding
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / self.target_size[-2]
        cell[:, 1] *= 2 / self.target_size[-1]

        results['coord'] = hr_coord
        results['cell'] = cell
        results['target_size'] = self.target_size

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'sample_quantity={self.sample_quantity}, '
                     f'scale={self.scale}, target_size={self.target_size}')
        return repr_str


class GenerateCoordinateAndCell2:
    """Generate coordinate and cell.
    Generate coordinate from the desired size of SR image.
        Train or val:
            1. Generate coordinate from GT.
            2. Reshape GT image to (HgWg, 3) and transpose to (3, HgWg).
                where `Hg` and `Wg` represent the height and width of GT.
        Test:
            Generate coordinate from LQ and scale or target_size.
    Then generate cell from coordinate.
    Args:
        sample_quantity (int): The quantity of samples in coordinates.
            To ensure that the GT tensors in a batch have the same dimensions.
            Default: None.
        scale (float): Scale of upsampling.
            Default: None.
        target_size (tuple[int]): Size of target image.
            Default: None.
    The priority of getting 'size of target image' is:
        1, results['gt'].shape[-2:]
        2, results['lq'].shape[-2:] * scale
        3, target_size
    """

    def __init__(self, sample_quantity=None, scale=None, scale1=None, target_size=None):
        self.sample_quantity = sample_quantity
        self.scale = scale
        self.scale1 = scale1
        self.target_size = target_size

    def __call__(self, results):
        """Call function.
        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.
                Require either in results:
                    1. 'lq' (tensor), whose shape is similar as (3, H, W).
                    2. 'gt' (tensor), whose shape is similar as (3, H, W).
                    3. None, the premise is
                        self.target_size and len(self.target_size) >= 2.
        Returns:
            dict: A dict containing the processed data and information.
                Reshape 'gt' to (-1, 3) and transpose to (3, -1) if 'gt'
                in results.
                Add 'coord' and 'cell'.
        """
        # generate hr_coord (and hr_rgb)
        if 'gt' in results:
            crop_hr = results['gt']
            # self.target_size = crop_hr.shape
            hr_rgb = crop_hr.contiguous().view(3, -1).permute(1, 0)
            results['gt'] = hr_rgb
            
            _, h_hr, w_hr = crop_hr.shape
            h_lr, w_lr = h_hr/self.scale, w_hr/self.scale
            self.target_size = (round(h_lr * self.scale1),
                                round(w_lr * self.scale1))

        # elif self.scale is not None and 'lq' in results:
        #     _, h_lr, w_lr = results['lq'].shape
        #     self.target_size = (round(h_lr * self.scale),
        #                         round(w_lr * self.scale))
        else:
            assert self.target_size is not None
            assert len(self.target_size) >= 2
        hr_coord = make_coord(self.target_size[-2:])

        if self.sample_quantity is not None and 'gt' in results:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_quantity, replace=False)
            hr_coord = hr_coord[sample_lst]
            results['gt'] = results['gt'][sample_lst]

        # Preparations for cell decoding
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / self.target_size[-2]
        cell[:, 1] *= 2 / self.target_size[-1]

        results['coord'] = hr_coord
        results['cell'] = cell

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'sample_quantity={self.sample_quantity}, '
                     f'scale={self.scale}, target_size={self.target_size}')
        return repr_str
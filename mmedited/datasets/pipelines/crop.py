# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from torch.nn.modules.utils import _pair


class ModCrop:
    """Mod crop gt images, used during testing.
    Required keys are "scale", "lq", "gt", added or modified keys are "lq, gt".
    """

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, results):
        """Call function.
        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.
        Returns:
            dict: A dict containing the processed data and information.
        """
        if 'gt' in results:
            imgs = results['gt'].copy()
            if imgs[0].ndim in [2, 3]:
                h, w = imgs[0].shape[0], imgs[0].shape[1]
                h_remainder, w_remainder = h % self.scale, w % self.scale
                results['gt'] = [
                    img[:h - h_remainder, :w - w_remainder, ...] for img in imgs
                ]
            else:
                raise ValueError(f'Wrong img ndim: {imgs[0].ndim}.')
        
        if 'lq' in results:
            imgs = results['lq'].copy()
            if imgs[0].ndim in [2, 3]:
                h, w = imgs[0].shape[0], imgs[0].shape[1]
                h_remainder, w_remainder = h % self.scale, w % self.scale
                results['lq'] = [
                    img[:h - h_remainder, :w - w_remainder, ...] for img in imgs
                ]
            else:
                raise ValueError(f'Wrong img ndim: {imgs[0].ndim}.')

        return results


# @PIPELINES.register_module()
class RandomCrop:
    """random crop.

    It crops gt images with corresponding locations.
    It also supports accepting gt list.
    Required keys are "scale" and "gt",
    added or modified keys is "gt".

    Args:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        gt_is_list = isinstance(results["gt"], list)
        if not gt_is_list:
            results["gt"] = [results["gt"]]

        h_gt, w_gt, _ = results["gt"][0].shape

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_gt - self.gt_patch_size + 1)
        left = np.random.randint(w_gt - self.gt_patch_size + 1)
        # crop corresponding gt patch
        results["gt"] = [
            v[top : top + self.gt_patch_size, left : left + self.gt_patch_size, ...]
            for v in results["gt"]
        ]

        if not gt_is_list:
            results["gt"] = results["gt"][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(gt_patch_size={self.gt_patch_size})"
        return repr_str


class ContinuousPairedRandomCrop:
    """Paried random crop.
    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".
    Args:
        lq_patch_size (int): cropped lq patch size.
    """

    def __init__(self, lq_patch_size, scale_min, scale_max):
        self.lq_patch_size = lq_patch_size
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, results):
        """Call function.
        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.
        Returns:
            dict: A dict containing the processed data and information.
        """
        scale = np.random.uniform(self.scale_min, self.scale_max)

        lq_patch_size = self.lq_patch_size
        gt_patch_size = round(lq_patch_size * scale)

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        # if h_gt != h_lq * scale or w_gt != w_lq * scale:
        #     raise ValueError(
        #         f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
        #         f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size ',
                f'({lq_patch_size}, {lq_patch_size}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_lq - lq_patch_size + 1)
        left = np.random.randint(w_lq - lq_patch_size + 1)
        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + gt_patch_size,
              left_gt:left_gt + gt_patch_size, ...] for v in results['gt']
        ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'
        return repr_str


class PairedRandomCropwScale:
    """Paried random crop.
    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".
    Args:
        lq_patch_size (int): cropped lq patch size.
    """

    def __init__(self, lq_patch_size):
        self.lq_patch_size = lq_patch_size

    def __call__(self, results):
        """Call function.
        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.
        Returns:
            dict: A dict containing the processed data and information.
        """

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape
        
        scale = h_gt / h_lq

        if h_gt / h_lq != w_gt / w_lq:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x '
                f'multiplication of LQ ({h_lq}, {w_lq}).')

        lq_patch_size = self.lq_patch_size
        self.gt_patch_size = int(lq_patch_size * scale)

        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                f'({lq_patch_size}, {lq_patch_size}). Please check ')

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_lq - lq_patch_size + 1)
        left = np.random.randint(w_lq - lq_patch_size + 1)
        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'
        return repr_str
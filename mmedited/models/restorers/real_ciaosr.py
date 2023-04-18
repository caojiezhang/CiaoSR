# Copyright (c) OpenMMLab. All rights reserved.
import math
import numbers
import os.path as osp

import mmcv
import torch
from copy import deepcopy
from mmcv.parallel import is_module_wrapper
from mmedit.models.common import set_requires_grad

from mmedit.core import tensor2img
# from mmedit.models.restorers.basic_restorer import BasicRestorer
from mmedit.models.restorers.srgan import SRGAN
from mmedit.datasets.pipelines.utils import make_coord

import socket
in_fbcode = True if 'fb' in socket.gethostname() or 'facebook' in socket.gethostname() else False

if in_fbcode:
    from iopath.common.file_io import g_pathmgr
    from on_device_ai.cvg.interns.projects.restoration_transformer.mmedited.models.restorers.basic_restorer import BasicRestorer
else:
    from mmedited.models.restorers.basic_restorer import BasicRestorer
import pdb


class RealCiaoSR(SRGAN):
    """CiaoSR model for single image super-resolution.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        rgb_mean (tuple[float]): Data mean.
            Default: (0.5, 0.5, 0.5).
        rgb_std (tuple[float]): Data std.
            Default: (0.5, 0.5, 0.5).
        train_cfg (dict): Config for train. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss=None,
                 perceptual_loss=None,
                 discriminator=None,
                 gan_loss=None,
                 rgb_mean=(0.5, 0.5, 0.5),
                 rgb_std=(0.5, 0.5, 0.5),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 is_use_sharpened_gt_in_pixel=False,
                 is_use_sharpened_gt_in_percep=False,
                 is_use_sharpened_gt_in_gan=False,
                 is_use_ema=True,
                 ):
        super().__init__(
            generator,
            discriminator,
            gan_loss,
            pixel_loss,
            perceptual_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        # norm
        rgb_mean = torch.FloatTensor(rgb_mean)
        rgb_std = torch.FloatTensor(rgb_std)
        self.lq_mean = rgb_mean.view(1, -1, 1, 1)
        self.lq_std = rgb_std.view(1, -1, 1, 1)
        self.gt_mean = rgb_mean.view(1, 1, -1)
        self.gt_std = rgb_std.view(1, 1, -1)

        # real sr
        self.is_use_sharpened_gt_in_pixel = is_use_sharpened_gt_in_pixel
        self.is_use_sharpened_gt_in_percep = is_use_sharpened_gt_in_percep
        self.is_use_sharpened_gt_in_gan = is_use_sharpened_gt_in_gan
        self.is_use_ema = is_use_ema
        if is_use_ema:
            self.generator_ema = deepcopy(self.generator)
        else:
            self.generator_ema = None

        del self.step_counter
        self.register_buffer('step_counter', torch.zeros(1))

        if train_cfg is not None:  # used for initializing from ema model
            self.start_iter = train_cfg.get('start_iter', -1)
        else:
            self.start_iter = -1

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data, which requires
                'coord', 'lq', 'gt', 'cell'
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output, which includes:
                log_vars, num_samples, results (lq, gt and pred).

        """
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']
        gt_unsharp = data_batch['gt_unsharp']
        coord = data_batch['coord']
        cell = data_batch['cell']
        
        # during initialization, load weights from the ema model
        if (self.step_counter == self.start_iter
                and self.generator_ema is not None):
            if is_module_wrapper(self.generator):
                self.generator.module.load_state_dict(
                    self.generator_ema.module.state_dict())
            else:
                self.generator.load_state_dict(self.generator_ema.state_dict())
        
        # norm
        self.lq_mean = self.lq_mean.to(lq)
        self.lq_std = self.lq_std.to(lq)
        self.gt_mean = self.gt_mean.to(gt)
        self.gt_std = self.gt_std.to(gt)
        lq = (lq - self.lq_mean) / self.lq_std
        gt = (gt - self.gt_mean) / self.gt_std
        gt_unsharp = (gt_unsharp - self.gt_mean) / self.gt_std

        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        if self.is_use_sharpened_gt_in_pixel:
            gt_pixel = gt_unsharp  
        if self.is_use_sharpened_gt_in_percep:
            gt_percep = gt_unsharp  
        if self.is_use_sharpened_gt_in_gan:
            gt_gan = gt_unsharp 

        # generator
        pred = self.generator(lq, coord, cell)
        _, ns, _ = pred.shape
        W, H = int(ns**0.5), int(ns**0.5)  #int(gt_size[-2]), int(gt_size[-1])

        # loss
        losses = dict()
        log_vars = dict()

        # no updates to discriminator parameters.
        if self.gan_loss:
            set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
            and self.step_counter >= self.disc_init_steps):
            
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(pred, gt_pixel)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(pred.view(-1, W, H, 3).permute(0, 3, 1, 2), 
                                                gt_percep.view(-1, W, H, 3).permute(0, 3, 1, 2))
                # loss_percep, loss_style = self.perceptual_loss(pred.view(-1, int(ns**0.5), int(ns**0.5), 3).permute(0, 3, 1, 2), 
                #                                 gt_percep.view(-1, int(ns**0.5), int(ns**0.5), 3).permute(0, 3, 1, 2))
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # gan loss for generator
            if self.gan_loss:
                fake_g_pred = self.discriminator(pred.view(-1, W, H, 3).permute(0, 3, 1, 2))
                # fake_g_pred = self.discriminator(pred.view(-1, int(ns**0.5), int(ns**0.5), 3).permute(0, 3, 1, 2))
                losses['loss_gan'] = self.gan_loss(
                    fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        else:
            losses['loss_pix'] = self.pixel_loss(pred, gt)
        
        # # parse loss
        # loss_g, log_vars_g = self.parse_losses(losses)
        # log_vars.update(log_vars_g)

        # # optimize
        # optimizer['generator'].zero_grad()
        # loss_g.backward()
        # optimizer['generator'].step()

        # discriminator
        if self.gan_loss:
            set_requires_grad(self.discriminator, True)
            # real
            real_d_pred = self.discriminator(gt_gan.view(-1, W, H, 3).permute(0, 3, 1, 2))
            # real_d_pred = self.discriminator(gt_gan.view(-1, int(ns**0.5), int(ns**0.5), 3).permute(0, 3, 1, 2))
            loss_d_real = self.gan_loss(
                real_d_pred, target_is_real=True, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_real=loss_d_real))
            optimizer['discriminator'].zero_grad()
            loss_d.backward()
            log_vars.update(log_vars_d)
            # fake
            fake_d_pred = self.discriminator(pred.view(-1, W, H, 3).permute(0, 3, 1, 2).detach())
            # fake_d_pred = self.discriminator(pred.view(-1, int(ns**0.5), int(ns**0.5), 3).permute(0, 3, 1, 2).detach())
            loss_d_fake = self.gan_loss(
                fake_d_pred, target_is_real=False, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_fake=loss_d_fake))
            loss_d.backward()
            log_vars.update(log_vars_d)

            optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=pred.cpu()))

        return outputs

    def forward_test(self,
                     lq,
                     gt,
                     coord,
                     cell,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ image.
            gt (Tensor): GT image.
            coord (Tensor): Coord tensor.
            cell (Tensor): Cell tensor.
            meta (list[dict]): Meta data, such as path of GT file.
                Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results, which contain either key(s)
                1. 'eval_result'.
                2. 'lq', 'pred'.
                3. 'lq', 'pred', 'gt'.
        """

        # norm
        self.lq_mean = self.lq_mean.to(lq)
        self.lq_std = self.lq_std.to(lq)
        lq = (lq - self.lq_mean) / self.lq_std

        _model = self.generator_ema if self.is_use_ema else self.generator

        # generator
        with torch.no_grad():
            
            if self.test_cfg.get('tile', None):
                pred = self.clip_test(lq, _model)
            else:
                pred = _model(lq, coord, cell, test_mode=True)

            self.gt_mean = self.gt_mean.to(pred)
            self.gt_std = self.gt_std.to(pred)
            pred = pred * self.gt_std + self.gt_mean
            pred.clamp_(0, 1)

        # reshape for eval
        ih, iw = lq.shape[-2:]
        s = math.sqrt(coord.shape[1] / (ih * iw))
        shape = [lq.shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        if gt is not None:
            gt = gt.view(*shape).permute(0, 3, 1, 2).contiguous()

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(pred, gt))
        else:
            results = dict(lq=lq.cpu(), output=pred.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:

            if 'gt_path' in meta[0]:
                gt_path = meta[0]['gt_path']
                folder_name = osp.splitext(osp.basename(gt_path))[0]
            else:
                lq_path = meta[0]['lq_path']
                folder_name = osp.splitext(osp.basename(lq_path))[0]
            
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(pred), save_path)

        return results

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """

        self.generator.init_weights(pretrained, strict)

    def clip_test(self, img_lq, model):
        
        sf = self.test_cfg.get('scale', None)
        b, c, h, w = img_lq.size()

        tile = min(self.test_cfg.get('tile', None), h, w)
        tile_overlap = self.test_cfg.get('tile_overlap', None)
        
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                
                target_size = (round(in_patch.shape[-2]*sf), 
                               round(in_patch.shape[-1]*sf))
                # pdb.set_trace()
                hr_coord = make_coord(target_size[-2:]).unsqueeze(0).cuda()
                cell = torch.ones_like(hr_coord).cuda()
                cell[:, :, 0] *= 2 / target_size[-2]
                cell[:, :, 1] *= 2 / target_size[-1]

                out_patch = model(in_patch, hr_coord, cell, test_mode=True) 
                
                ih, iw = in_patch.shape[-2:]
                shape = [in_patch.shape[0], round(ih * sf), round(iw * sf), 3]
                out_patch = out_patch.view(*shape).permute(0, 3, 1, 2).contiguous()

                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)
        output = output.view(b, 3, -1).permute(0,2,1).contiguous()
        return output
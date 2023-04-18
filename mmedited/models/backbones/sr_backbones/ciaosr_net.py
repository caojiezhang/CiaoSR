import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init
from mmedit.datasets.pipelines.utils import make_coord
from mmedit.models.builder import build_backbone, build_component
from mmedit.utils import get_root_logger
from einops import repeat
import math
import numpy as np
import time
from mmedited.models.common.arch_csnln import CrossScaleAttention
import pdb


class LocalImplicitSRNet(nn.Module):
    """
    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 query_mlp,
                 key_mlp,
                 value_mlp,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 local_ensemble_coord=False,  #lec
                 imnet_k_type='direct', 
                 imnet_v_type='direct',
                 qkv_mlp=False,
                 res=False,
                 use_lte=False,
                 lte_type="sin_cos",
                 multifeat=False,
                 parallel=False,
                 deformable=False,
                 non_local_attn=False,
                 ensemble_type='attn',
                 use_itsrn=False,
                 last_mul_w=False,
                 last_cat_coor=False,
                 cat_nla_v=False,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.eval_bsize = eval_bsize
        self.local_size = local_size
        self.local_ensemble_coord = local_ensemble_coord
        self.imnet_k_type = imnet_k_type
        self.imnet_v_type = imnet_v_type
        self.qkv_mlp = qkv_mlp
        self.res = res
        self.use_lte = use_lte
        self.lte_type = lte_type
        self.multifeat = multifeat
        self.parallel = parallel
        self.deformable = deformable
        self.non_local_attn = non_local_attn
        self.ensemble_type = ensemble_type
        self.use_itsrn = use_itsrn
        self.last_mul_w = last_mul_w
        self.last_cat_coor = last_cat_coor
        self.cat_nla_v = cat_nla_v
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale

        # imnet
        self.encoder = build_backbone(encoder)
        if hasattr(self.encoder, 'mid_channels'):
            imnet_dim = self.encoder.mid_channels
        else:
            imnet_dim = self.encoder.embed_dim
        if self.feat_unfold:
            imnet_q['in_dim'] = imnet_dim * 9
            imnet_k['in_dim'] = imnet_k['out_dim'] = imnet_dim * 9
            imnet_v['in_dim'] = imnet_v['out_dim'] = imnet_dim * 9
        else:
            imnet_q['in_dim'] = imnet_dim
            imnet_k['in_dim'] = imnet_k['out_dim'] = imnet_dim
            imnet_v['in_dim'] = imnet_v['out_dim'] = imnet_dim

        if local_ensemble_coord:
            imnet_k['in_dim'] += 4
            if not self.use_lte:
                imnet_v['in_dim'] += 4

        if self.use_itsrn:
            imnet_v['in_dim'] = 4
            imnet_v['out_dim'] *= 3
        
        if self.non_local_attn:
            imnet_q['in_dim'] += imnet_dim*len(multi_scale)
            if self.cat_nla_v:
                imnet_v['in_dim'] += imnet_dim*len(multi_scale)
                imnet_v['out_dim'] += imnet_dim*len(multi_scale)

        if self.multifeat:
            imnet_q['in_dim'] *= 2

        if self.ensemble_type == 'concat':
            imnet_q['in_dim'] *= 4
        elif self.ensemble_type == 'itsrn':
            imnet_q['in_dim'] = 4
            imnet_q['out_dim'] = 1

        if self.last_mul_w:
            imnet_w_in_dim = imnet_q['in_dim']
            imnet_w_out_dim = imnet_q['in_dim'] * 3
            if self.last_cat_coor:
                imnet_w_in_dim += 8
            self.imnet_weight = nn.Sequential(
                                nn.Linear(imnet_w_in_dim, 255),
                                nn.ReLU(),
                                nn.Linear(255, 255),
                                nn.ReLU(),
                                nn.Linear(255, 255),
                                nn.ReLU(),
                                nn.Linear(255, 255),
                                nn.ReLU(),
                                nn.Linear(255, imnet_w_out_dim))

        self.imnet_q = build_component(imnet_q) #362243
        self.imnet_k = build_component(imnet_k) #494144
        self.imnet_v = build_component(imnet_v) #526976, total:1383363
        
        if self.qkv_mlp:
            query_mlp['in_dim'] = query_mlp['out_dim'] = imnet_dim
            key_mlp['in_dim'] = key_mlp['out_dim'] = imnet_dim
            value_mlp['in_dim'] = value_mlp['out_dim'] = imnet_dim

            self.query_mlp = build_component(query_mlp)
            self.key_mlp = build_component(key_mlp)
            self.value_mlp = build_component(value_mlp)

        if self.use_lte:
            self.coef = nn.Conv2d(imnet_dim, imnet_dim * 9, 3, padding=1)
            self.freq = nn.Conv2d(imnet_dim, imnet_dim * 9, 3, padding=1)
            if self.lte_type == "sin":
                self.coor_enc = nn.Linear(2, imnet_dim * 9, bias=False)
                self.phase = nn.Linear(2, imnet_dim * 9, bias=False) 
            elif self.lte_type == "sin1":
                self.phase = nn.Linear(2, imnet_dim * 9, bias=False) 
                self.freq = nn.Conv2d(imnet_dim, imnet_dim * 9 *2, 3, padding=1) #
            elif self.lte_type == "sin2":
                self.coor_enc = nn.Linear(imnet_dim * 9+2, imnet_dim * 9, bias=False)  
                self.phase = nn.Linear(2, imnet_dim * 9, bias=False) 
            elif self.lte_type == "sin_cos":
                self.phase = nn.Linear(2, imnet_dim * 9//2, bias=False)   
            elif self.lte_type == "sin_cos_dif":
                self.phase = nn.Linear(2, imnet_dim * 9, bias=False)   
            elif self.lte_type == "2sin_cos":
                self.coef2 = nn.Conv2d(imnet_dim, imnet_dim * 9, 3, padding=1)
                self.phase = nn.Linear(2, imnet_dim * 9//2, bias=False)   
                self.phase2 = nn.Linear(2, imnet_dim * 9//2, bias=False)   
            elif self.lte_type == "2sin_cos_mlp":
                self.coef2 = nn.Conv2d(imnet_dim, imnet_dim * 9, 3, padding=1)
                self.phase = nn.Linear(2, imnet_dim * 9//2, bias=False)   
                self.phase2 = nn.Linear(2, imnet_dim * 9//2, bias=False)   
                self.freq_mlp = nn.Linear(imnet_dim * 9, imnet_dim * 9, bias=False)   

        if self.deformable:
            self.conv_offset  = nn.Sequential(
                                nn.Conv2d(imnet_dim, imnet_dim, 3, 1, 1),
                                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                nn.Conv2d(imnet_dim, imnet_dim, 3, 1, 1),
                                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                nn.Conv2d(imnet_dim, 2 * local_size * local_size, 3, 1, 1))
            # nn.init.constant_(self.conv_offset.weight, 0.1)
            # nn.init.constant_(self.conv_offset.bias, 0.)
            constant_init(self.conv_offset[-1], val=0.0001, bias=0)

            # self.conv_mask = nn.Sequential(
            #                     nn.Conv2d(imnet_dim, imnet_dim, 3, 1, 1),
            #                     nn.LeakyReLU(negative_slope=0.1, inplace=True),
            #                     nn.Conv2d(imnet_dim, imnet_dim, 3, 1, 1),
            #                     nn.LeakyReLU(negative_slope=0.1, inplace=True),
            #                     nn.Conv2d(imnet_dim, 2 * local_size * local_size, 3, 1, 1))
            # constant_init(self.conv_mask[-1], val=0, bias=0)
        
        if self.non_local_attn:
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)    #45251

        if self.use_itsrn:
            self.Score_itsrn = nn.Sequential(nn.Linear(2, 255),
                                             nn.GELU(), 
                                             nn.Linear(255, 1))
            

    def forward(self, x, coord, cell, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        # start = time.time()
        feature = self.gen_feature(x)

        if self.eval_bsize is None or not test_mode:
            pred = self.query_rgb(feature, coord, cell)
        else:
            pred = self.batched_predict(feature, coord, cell)

        # end = time.time()
        # running_time = end-start
        # print('time cost : %.5f sec' %running_time)
        # pdb.set_trace()

        if self.res:
            pred += F.grid_sample(x, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return pred


    def query_rgb(self, features, coord, scale=None):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """

        res_features = []
        for feature in features:
            
            B, C, H, W = feature.shape      #[16, 64, 48, 48]

            if self.qkv_mlp:
                feat_flat = feature.view(B, C, -1).permute(0, 2, 1).contiguous()        #[16, 2304, 64]
                feat_q = self.query_mlp(feat_flat).permute(0, 2, 1).view(B, C, H, W)    #[16, 64, 48, 48]
                feat_k = self.key_mlp(feat_flat).permute(0, 2, 1).view(B, C, H, W)      #[16, 64, 48, 48]
                feat_v = self.value_mlp(feat_flat).permute(0, 2, 1).view(B, C, H, W)    #[16, 64, 48, 48]

            if self.feat_unfold:
                if self.qkv_mlp:
                    feat_q = F.unfold(feat_q, 3, padding=1).view(B, C*9, H, W)          #[16, 576, 48, 48]
                    feat_k = F.unfold(feat_k, 3, padding=1).view(B, C*9, H, W)          #[16, 576, 48, 48]
                    feat_v = F.unfold(feat_v, 3, padding=1).view(B, C*9, H, W)          #[16, 576, 48, 48]
                else:
                    feat_q = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
                    feat_k = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
                    if self.non_local_attn:
                        # feat_v = self.cs_attn(feature)  # [16, 64, 48, 48]
                        # feat_v = F.unfold(feat_v, 3, padding=1).view(B, C*9, H, W)      #[16, 576, 48, 48]
                        non_local_feat_v = self.cs_attn(feature)  # [16, 64, 48, 48]
                        feat_v = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)      #[16, 576, 48, 48]
                        
                        if self.cat_nla_v:
                            feat_v = torch.cat([feat_v, non_local_feat_v], dim=1)       #[16, 576+64, 48, 48]
                    else:
                        feat_v = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)     #[16, 576, 48, 48]
            else:
                feat_q = feat_k = feat_v = feature

            if self.use_lte:
                coef = self.coef(feature)   #[16, 576, 48, 48]
                freq = self.freq(feature)   #[16, 576, 48, 48]
                if "2sin_cos" in self.lte_type:
                    coef2 = self.coef2(feature)   #[16, 576, 48, 48]

            # query
            query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest', 
                        align_corners=False).permute(0, 3, 2, 1).contiguous()    #[16, 2304, 1, 576]

            if self.non_local_attn:
                if not self.cat_nla_v:
                    non_local_feat_v = F.grid_sample(non_local_feat_v, coord.flip(-1).unsqueeze(1), mode='nearest', 
                            align_corners=False).permute(0, 3, 2, 1).contiguous()    #[16, 2304, 1, 576]

            feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
                            .unsqueeze(0).expand(B, 2, *feature.shape[-2:])         #[16, 2, 48, 48]
            feat_coord = feat_coord.to(coord)

            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                v_lst = [(i,j) for i in range(-1, 2, 4-self.local_size) for j in range(-1, 2, 4-self.local_size)]
            eps_shift = 1e-6
            preds_k, preds_v = [], []
            
            if not self.parallel:
                for v in v_lst:
                    vx, vy = v[0], v[1]
                    # project to LR field
                    tx = ((H - 1) / (1 - scale[:,0,0])).view(B,  1)     # [16, 1]
                    ty = ((W - 1) / (1 - scale[:,0,1])).view(B,  1)     # [16, 1]
                    rx = (2*abs(vx) -1) / tx if vx != 0 else 0          # [16, 1]
                    ry = (2*abs(vy) -1) / ty if vy != 0 else 0          # [16, 1]
                    
                    bs, q = coord.shape[:2]     
                    coord_ = coord.clone()  # [16, 2304, 2]
                    if vx != 0:
                        coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift  # [16, 2304]
                    if vy != 0:
                        coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift  # [16, 2304]
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                    # key and value
                    key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                        align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()       #[16, 2304, 576]
                    value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                        align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()       #[16, 2304, 576]

                    if self.local_ensemble_coord:
                        #Interpolate K to HR resolution
                        coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  #[16, 2304, 2]

                        Q, K = coord, coord_k   #[16, 2304, 2]
                        rel = Q - K             #[16, 2304, 2]
                        rel[:, :, 0] *= feature.shape[-2]   # without mul
                        rel[:, :, 1] *= feature.shape[-1]
                        inp = rel   #[16, 2304, 2]

                        scale_ = scale.clone()      #[16, 2304, 2]
                        scale_[:, :, 0] *= feature.shape[-2]
                        scale_[:, :, 1] *= feature.shape[-1]

                        if self.use_lte:
                            coef_v = F.grid_sample(coef, coord_.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [16, 2304, 576]
                            freq_v = F.grid_sample(freq, coord_.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [16, 2304, 576] or 576*2
                        
                            if self.lte_type == "sin":
                                freq_v = torch.mul(freq_v, self.coor_enc(inp))                  # [16, 2304, 576]
                                freq_v += self.phase(scale_.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 576]
                                freq_v = torch.sin(freq_v)                                      # [16, 2304, 576]
                                inp_v = torch.mul(coef_v, freq_v)

                            elif self.lte_type == "sin1":
                                freq_v = torch.stack(torch.split(freq_v, 2, dim=-1), dim=-1)    # [16, 2304, 2, 576]
                                freq_v = torch.mul(freq_v, rel.unsqueeze(-1))                   # [16, 2304, 2, 576]
                                freq_v = torch.sum(freq_v, dim=-2)                              # [16, 2304, 576] 
                                freq_v += self.phase(scale_.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 576]
                                freq_v = torch.sin(freq_v)                                      # [16, 2304, 576]
                                inp_v = torch.mul(coef_v, freq_v)

                            elif self.lte_type == "sin2":
                                freq_v = self.coor_enc(torch.cat([value, inp], dim=-1))           # [16, 2304, 576]
                                freq_v += self.phase(scale_.view((bs * q, -1))).view(bs, q, -1)   # [16, 2304, 576]
                                freq_v = torch.sin(freq_v)                                        # [16, 2304, 576]
                                inp_v = torch.mul(coef_v, freq_v)

                            elif self.lte_type == "sin_cos":
                                freq_v = torch.stack(torch.split(freq_v, 2, dim=-1), dim=-1)    # [16, 2304, 2, 288]
                                freq_v = torch.mul(freq_v, rel.unsqueeze(-1))                   # [16, 2304, 2, 288]
                                freq_v = torch.sum(freq_v, dim=-2)                              # [16, 2304, 288]
                                freq_v += self.phase(scale_.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 288]
                                freq_v = torch.cat((torch.cos(np.pi*freq_v), torch.sin(np.pi*freq_v)), dim=-1)  # [16, 2304, 576]
                                inp_v = torch.mul(coef_v, freq_v)   # [16, 2304, 576]
                            
                            elif self.lte_type == "sin_cos_dif":
                                freq_v = torch.stack(torch.split(freq_v, 2, dim=-1), dim=-1)    # [16, 2304, 2, 288]
                                freq_v = torch.mul(freq_v, rel.unsqueeze(-1)).view(bs, q, -1)   # [16, 2304, 576]
                                freq_v += self.phase(scale_.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 576]
                                mid_idx = freq_v.shape[2] // 2
                                freq_v = torch.cat((torch.cos(np.pi*freq_v[:, :, :mid_idx]), 
                                                    torch.sin(np.pi*freq_v[:, :, mid_idx:])), dim=-1)  # [16, 2304, 576]
                                inp_v = torch.mul(coef_v, freq_v)   # [16, 2304, 576]
                            
                            elif self.lte_type == "2sin_cos":
                                freq_v1 = torch.stack(torch.split(freq_v, 2, dim=-1), dim=-1)    # [16, 2304, 2, 288]
                                freq_v1 = torch.mul(freq_v1, rel.unsqueeze(-1))                   # [16, 2304, 2, 288]
                                freq_v1 = torch.sum(freq_v1, dim=-2)                              # [16, 2304, 288]
                                freq_v1 += self.phase(scale_.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 288]
                                freq_v1 = torch.cat((torch.cos(np.pi*freq_v1), torch.sin(np.pi*freq_v1)), dim=-1)  # [16, 2304, 576]
                                freq_v1 = torch.mul(coef_v, freq_v1)                              # [16, 2304, 576]

                                freq_v2 = torch.stack(torch.split(freq_v1, 2, dim=-1), dim=-1)    # [16, 2304, 2, 288]
                                freq_v2 = torch.mul(freq_v2, rel.unsqueeze(-1))                   # [16, 2304, 2, 288]
                                freq_v2 = torch.sum(freq_v2, dim=-2)                              # [16, 2304, 288]
                                freq_v2 += self.phase2(scale_.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 288]
                                freq_v2 = torch.cat((torch.cos(np.pi*freq_v2), torch.sin(np.pi*freq_v2)), dim=-1)  # [16, 2304, 576]
                                coef_v2 = F.grid_sample(coef2, coord_.flip(-1).unsqueeze(1),
                                            mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [16, 2304, 576]
                                inp_v = torch.mul(coef_v2, freq_v2)   # [16, 2304, 576]

                            elif self.lte_type == "2sin_cos_mlp":
                                freq_v1 = torch.stack(torch.split(freq_v, 2, dim=-1), dim=-1)    # [16, 2304, 2, 288]
                                freq_v1 = torch.mul(freq_v1, rel.unsqueeze(-1))                   # [16, 2304, 2, 288]
                                freq_v1 = torch.sum(freq_v1, dim=-2)                              # [16, 2304, 288]
                                freq_v1 += self.phase(scale_.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 288]
                                freq_v1 = torch.cat((torch.cos(np.pi*freq_v1), torch.sin(np.pi*freq_v1)), dim=-1)  # [16, 2304, 576]
                                freq_v1 = torch.mul(coef_v, freq_v1)                              # [16, 2304, 576]

                                freq_v2 = self.freq_mlp(freq_v1)                                  # [16, 2304, 576]
                                freq_v2 = torch.stack(torch.split(freq_v2, 2, dim=-1), dim=-1)    # [16, 2304, 2, 288]
                                freq_v2 = torch.mul(freq_v2, rel.unsqueeze(-1))                   # [16, 2304, 2, 288]
                                freq_v2 = torch.sum(freq_v2, dim=-2)                              # [16, 2304, 288]
                                freq_v2 += self.phase2(scale_.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 288]
                                freq_v2 = torch.cat((torch.cos(np.pi*freq_v2), torch.sin(np.pi*freq_v2)), dim=-1)  # [16, 2304, 576]
                                coef_v2 = F.grid_sample(coef2, coord_.flip(-1).unsqueeze(1),
                                            mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [16, 2304, 576]
                                inp_v = torch.mul(coef_v2, freq_v2)   # [16, 2304, 576]

                        elif self.use_itsrn:
                            inp_v = torch.cat([inp, scale_], dim=-1)   #[16, 2304, 4]
                        else:
                            inp_v = torch.cat([value, inp, scale_], dim=-1)   #[16, 2304, 580]

                        inp_k = torch.cat([key, inp, scale_], dim=-1)   #[16, 2304, 580]
                        
                    else:
                        inp_k = key
                        inp_v = value

                    inp_k = inp_k.contiguous().view(bs * q, -1)
                    inp_v = inp_v.contiguous().view(bs * q, -1)

                    if self.imnet_k_type == 'direct':
                        pred_k = self.imnet_k(inp_k).view(bs, q, -1)     #[16, 2304, 576]
                    elif self.imnet_k_type == 'mul_w':
                        weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous() #[16, 2304, 576]
                        pred_k = (key * weight_k).view(bs, q, -1)       #[16, 2304, 576]
                    
                    if self.imnet_v_type == 'direct':
                        pred_v = self.imnet_v(inp_v).view(bs, q, -1)   #[16, 2304, 576]
                    elif self.imnet_v_type == 'mul_w':
                        weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()   #[16, 2304, 576]
                        pred_v = (value * weight_v).view(bs, q, -1)       #[16, 2304, 576]
                    elif self.imnet_v_type == 'itsrn':
                        score = repeat(self.Score_itsrn(rel.view(bs * q, -1)).view(bs, q, -1),
                                            'b q c -> b q (repeat c)', repeat=3) # [16, 2304, 3]
                        weight = self.imnet_v(inp_v).view(bs * q, feat_v.shape[1], 3)   # [36864, 576, 3]
                        pred_v = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)  # [16, 2304, 3]
                        pred_v += score     # [16, 2304, 3]

                    preds_v.append(pred_v)
                    preds_k.append(pred_k)

                preds_k = torch.stack(preds_k, dim=-1)  # [16, 2304, 576, 4]
                preds_v = torch.stack(preds_v, dim=-2)  # [16, 2304, 4, 576]

            else:
                v_lst_ = torch.tensor(v_lst).unsqueeze(0).cuda()    # [1, 4, 2]
                vx, vy = v_lst_[:, :, 0:1], v_lst_[:, :, 1:]        # [1, 4, 1]

                # project to LR field
                tx = ((H - 1) / (1 - scale[:,0,0])).view(B, 1, 1).repeat(1, len(v_lst), 1)     # [16, 4, 1]
                ty = ((W - 1) / (1 - scale[:,0,1])).view(B, 1, 1).repeat(1, len(v_lst), 1)     # [16, 4, 1]
                rx = (vx!=0) * (2*abs(vx) -1) / tx          # [16, 4, 1]
                ry = (vy!=0) * (2*abs(vy) -1) / ty          # [16, 4, 1]

                bs, q = coord.shape[:2]  
                coord_ = coord.clone().unsqueeze(1).repeat(1, len(v_lst), 1, 1)   # [16, 4, 2304, 2]
                coord_[:, :, :, 0] += (vx!=0) * (vx /abs(vx+(vx==0)*eps_shift) * rx + eps_shift)    # [16, 4, 2304]
                coord_[:, :, :, 1] += (vy!=0) * (vy /abs(vy+(vx==0)*eps_shift) * ry + eps_shift)    # [16, 4, 2304]
                if self.deformable:
                    offset_min, offset_max = 2*torch.min(rx.min(), ry.min()), 2*torch.max(rx.max(), ry.max())
                    offset = 10*torch.tanh(self.conv_offset(feature))      # 2*offset_min* [16, 8, 48, 48] .clamp(-offset_min, offset_max)
                    # print(offset.min(), offset.max())
                    offset = offset.view(bs, self.local_size**2, 2, H, W)
                    offset = offset.view(bs*self.local_size**2, 2, H, W)
                    offset = F.grid_sample(offset, coord_.view(bs*len(v_lst), q, 2).flip(-1).unsqueeze(1), mode='nearest', 
                                align_corners=False)[:, :, 0, :].permute(0, 2, 1).view(bs, len(v_lst), q, -1).contiguous()  # [16, 4, 2304, 2]
                    # mask = 2. * torch.sigmoid(self.conv_mask(feature))
                    coord_ += offset   # [16, 4, 2304, 2]
                    
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                coord_ = coord_.view(bs*len(v_lst), q, 2)   # [16*4, 2304, 2]

                # key and value
                feat_k_tmp = feat_k.unsqueeze(1).repeat(1, len(v_lst), 1, 1, 1).view(bs*len(v_lst), -1, H, W)  # [64, 576, 48, 48]
                feat_v_tmp = feat_v.unsqueeze(1).repeat(1, len(v_lst), 1, 1, 1).view(bs*len(v_lst), -1, H, W)  # [64, 576, 48, 48]
                key = F.grid_sample(feat_k_tmp, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                                align_corners=False)[:, :, 0, :].permute(0, 2, 1).view(bs, len(v_lst), q, -1).contiguous()  # [16, 4, 2304, 576]
                value = F.grid_sample(feat_v_tmp, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                                align_corners=False)[:, :, 0, :].permute(0, 2, 1).view(bs, len(v_lst), q, -1).contiguous()  # [16, 4, 2304, 576]
                
                if self.local_ensemble_coord:
                    #Interpolate K to HR resolution
                    feat_coord_tmp = feat_coord.unsqueeze(1).repeat(1, len(v_lst), 1, 1, 1).view(bs*len(v_lst), -1, H, W) # [16*4, 2, 48, 48]
                    coord_k = F.grid_sample(feat_coord_tmp, coord_.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  #[16*4, 2304, 2]
                    
                    Q, K = coord.unsqueeze(1), coord_k.view(bs, len(v_lst), q, 2)  # [16, 1, 2304, 2], [16, 4, 2304, 2]
                    rel = Q - K                         # [16, 4, 2304, 2]
                    rel[:, :, :, 0] *= feature.shape[-2]
                    rel[:, :, :, 1] *= feature.shape[-1]
                    inp = rel   #[16, 4, 2304, 2]
                    
                    scale_ = scale.clone()      #[16, 2304, 2]
                    scale_[:, :, 0] *= feature.shape[-2]
                    scale_[:, :, 1] *= feature.shape[-1]
                    scale_ = scale_.unsqueeze(1).repeat(1, len(v_lst), 1, 1)   #[16, 4, 2304, 2]

                    inp_v = torch.cat([value, inp, scale_], dim=-1)   #[16, 4, 2304, 580]
                    inp_k = torch.cat([key, inp, scale_], dim=-1)   #[16, 4, 2304, 580]
                else:
                    inp_k = key
                    inp_v = value

                if self.imnet_k_type == 'direct':
                    preds_k = self.imnet_k(inp_k)       # [16, 4, 2304, 576]
                elif self.imnet_k_type == 'mul_w':
                    weight_k = self.imnet_k(inp_k)      # [16, 4, 2304, 576]
                    preds_k = (key * weight_k)          # [16, 4, 2304, 576]
                preds_k = preds_k.permute(0, 2, 3, 1)   # [16, 2304, 576, 4]

                if self.imnet_v_type == 'direct':
                    preds_v = self.imnet_v(inp_v)       # [16, 4, 2304, 576]
                elif self.imnet_v_type == 'mul_w':
                    weight_v = self.imnet_v(inp_v)      # [16, 4, 2304, 576]
                    preds_v = (value * weight_v)        # [16, 4, 2304, 576]
                preds_v = preds_v.permute(0, 2, 1, 3)   # [16, 2304, 4, 576]
            
            if self.ensemble_type == 'attn':
                attn = (query @ preds_k)                # [16, 2304, 1, 4]
                # print(attn.softmax(dim=-1)[0,10,0,:])
                x = ((attn/self.softmax_scale).softmax(dim=-1) @ preds_v)    # [16, 2304, 1, 576]
                x = x.view(bs*q, -1)       # [16*2304, 576]
            elif self.ensemble_type == 'concat':
                x = preds_v
                x = x.view(bs*q, -1)       # [16*2304, 3]
            elif self.ensemble_type == 'itsrn':
                x = preds_v.permute(0, 1, 3, 2).contiguous()
                x = x.view(bs*q, -1, len(v_lst))       # [16*2304, 3]

            if self.last_mul_w:
                if self.last_cat_coor:
                    coord_q = F.grid_sample(feat_coord, coord.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  #[16, 2304, 2]

                    Q, K = coord, coord_q
                    rel = Q - K     #[16, 2304, 2]
                    rel[:, :, 0] *= feature.shape[-2]
                    rel[:, :, 1] *= feature.shape[-1]
                    inp_x = torch.cat([x.view(bs, q, -1), attn.view(bs, q, -1), rel, scale_], dim=-1)   #[16, 2304, 8]
                    weight = self.imnet_weight(inp_x.view(bs * q, -1)).view(bs * q, -1, 3)         #[36864, 576, 3]
                    x = torch.bmm(x.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)    #[16, 2304, 3]
                else:
                    weight = self.imnet_weight(x.view(bs * q, -1)).view(bs * q, -1, 3)         #[36864, 576, 3]
                    x = torch.bmm(x.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)    #[16, 2304, 3]

            res_features.append(x) 

            if self.non_local_attn:
                if not self.cat_nla_v:
                    res_features.append(non_local_feat_v.view(bs*q, -1))

        result = torch.cat(res_features, dim=-1)  # [16, 2304, 576x2]

        if not self.last_mul_w:
            result = self.imnet_q(result)      # [16, 2304, 3]
        
        result = result.view(bs, q, -1)

        return result

    def batched_predict(self, x, coord, cell):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad(): 
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + self.eval_bsize, n)
                pred = self.query_rgb(x, coord[:, left:right, :],
                                      cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred

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
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class LocalImplicitSRRDN(LocalImplicitSRNet):
    """ITSRN net based on RDN.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feat unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 query_mlp=None,
                 key_mlp=None,
                 value_mlp=None,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 local_ensemble_coord=False,
                 imnet_k_type='direct',
                 imnet_v_type='direct',
                 qkv_mlp=False,
                 res=False,
                 use_lte=False,
                 lte_type="sin_cos",
                 multifeat=False,
                 parallel=False,
                 deformable=False,
                 non_local_attn=False,
                 ensemble_type='attn',
                 use_itsrn=False,
                 last_mul_w=False,
                 last_cat_coor=False,
                 cat_nla_v=False,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__(
            encoder=encoder,
            imnet_q=imnet_q,
            imnet_k=imnet_k,
            imnet_v=imnet_v,
            query_mlp=query_mlp,
            key_mlp=key_mlp,
            value_mlp=value_mlp,
            local_size=local_size,
            feat_unfold=feat_unfold,
            eval_bsize=eval_bsize,
            local_ensemble_coord=local_ensemble_coord,
            imnet_k_type=imnet_k_type,
            imnet_v_type=imnet_v_type,
            qkv_mlp=qkv_mlp,
            res=res,
            use_lte=use_lte,
            lte_type=lte_type,
            multifeat=multifeat,
            parallel=parallel,
            deformable=deformable,
            non_local_attn=non_local_attn,
            ensemble_type=ensemble_type,
            use_itsrn=use_itsrn,
            last_mul_w=last_mul_w,
            last_cat_coor=last_cat_coor,
            cat_nla_v=cat_nla_v,
            multi_scale=multi_scale,
            softmax_scale=softmax_scale,
            )

        self.multifeat = multifeat

        self.sfe1 = self.encoder.sfe1
        self.sfe2 = self.encoder.sfe2
        self.rdbs = self.encoder.rdbs
        self.gff = self.encoder.gff
        self.num_blocks = self.encoder.num_blocks
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1

        if self.multifeat:
            idx = len(local_features) // 2
            return [local_features[idx], x]
        else:
            return [x]


class LocalImplicitSREDSR(LocalImplicitSRNet):
    """LocalImplicitSR based on EDSR.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 query_mlp=None,
                 key_mlp=None,
                 value_mlp=None,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 local_ensemble_coord=False,
                 imnet_k_type='direct',
                 imnet_v_type='direct',
                 qkv_mlp=False,
                 res=False,
                 use_lte=False,
                 lte_type="sin_cos",
                 multifeat=False,
                 parallel=False,
                 deformable=False,
                 non_local_attn=False,
                 ensemble_type='attn',
                 use_itsrn=False,
                 last_mul_w=False,
                 last_cat_coor=False,
                 cat_nla_v=False,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__(
            encoder=encoder,
            imnet_q=imnet_q,
            imnet_k=imnet_k,
            imnet_v=imnet_v,
            query_mlp=query_mlp,
            key_mlp=key_mlp,
            value_mlp=value_mlp,
            local_size=local_size,
            feat_unfold=feat_unfold,
            eval_bsize=eval_bsize,
            local_ensemble_coord=local_ensemble_coord,
            imnet_k_type=imnet_k_type,
            imnet_v_type=imnet_v_type,
            qkv_mlp=qkv_mlp,
            res=res,
            use_lte=use_lte,
            lte_type=lte_type,
            multifeat=multifeat,
            parallel=parallel,
            deformable=deformable,
            non_local_attn=non_local_attn,
            ensemble_type=ensemble_type,
            use_itsrn=use_itsrn,
            last_mul_w=last_mul_w,
            last_cat_coor=last_cat_coor,
            cat_nla_v=cat_nla_v,
            multi_scale=multi_scale,
            softmax_scale=softmax_scale,
            )

        self.conv_first = self.encoder.conv_first
        self.body = self.encoder.body
        self.conv_after_body = self.encoder.conv_after_body
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res += x

        return [res]


class LocalImplicitSRSWINIR(LocalImplicitSRNet):
    """ITSRN net based on EDSR.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 window_size,
                 encoder,
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 query_mlp=None,
                 key_mlp=None,
                 value_mlp=None,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 local_ensemble_coord=False,
                 imnet_k_type='direct',
                 imnet_v_type='direct',
                 qkv_mlp=False,
                 res=False,
                 use_lte=False,
                 lte_type="sin_cos",
                 multifeat=False,
                 parallel=False,
                 deformable=False,
                 non_local_attn=False,
                 ensemble_type='attn',
                 use_itsrn=False,
                 last_mul_w=False,
                 last_cat_coor=False,
                 cat_nla_v=False,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__(
            encoder=encoder,
            imnet_q=imnet_q,
            imnet_k=imnet_k,
            imnet_v=imnet_v,
            query_mlp=query_mlp,
            key_mlp=key_mlp,
            value_mlp=value_mlp,
            local_size=local_size,
            feat_unfold=feat_unfold,
            eval_bsize=eval_bsize,
            local_ensemble_coord=local_ensemble_coord,
            imnet_k_type=imnet_k_type,
            imnet_v_type=imnet_v_type,
            qkv_mlp=qkv_mlp,
            res=res,
            use_lte=use_lte,
            lte_type=lte_type,
            multifeat=multifeat,
            parallel=parallel,
            deformable=deformable,
            non_local_attn=non_local_attn,
            ensemble_type=ensemble_type,
            use_itsrn=use_itsrn,
            last_mul_w=last_mul_w,
            last_cat_coor=last_cat_coor,
            cat_nla_v=cat_nla_v,
            multi_scale=multi_scale,
            softmax_scale=softmax_scale,
            )

        self.window_size = window_size

        self.conv_first = self.encoder.conv_first

        # body
        self.patch_embed = self.encoder.patch_embed
        self.pos_drop = self.encoder.pos_drop
        self.layers = self.encoder.layers
        self.norm = self.encoder.norm
        self.patch_unembed = self.encoder.patch_unembed

        self.conv_after_body = self.encoder.conv_after_body

        # self.apply(self._init_weights)

        del self.encoder

    def forward_features(self, x):

        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def gen_feature(self, img):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % self.window_size != 0:
            mod_pad_h = self.window_size - h % self.window_size
        if w % self.window_size != 0:
            mod_pad_w = self.window_size - w % self.window_size
        img_pad = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        x = self.conv_first(img_pad)
        res = self.forward_features(x)
        res = self.conv_after_body(res)
        res += x

        _, _, h, w = res.size()
        res = res[:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]

        return [res]


if __name__ == "__main__":
    encoder=dict(
            type='RDN',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=4,
            num_layers=8,
            channel_growth=64),
    imnet_q=dict(
        type='MLPRefiner',
        in_dim=4,
        out_dim=3,
        hidden_list=[256, 256, 256, 256]),
    imnet_k=dict(
        type='MLPRefiner',
        in_dim=64,
        out_dim=64,
        hidden_list=[256, 256, 256, 256]),
    imnet_v=dict(
        type='MLPRefiner',
        in_dim=64,
        out_dim=64,
        hidden_list=[256, 256, 256, 256])

    model1 = LocalImplicitSRRDN(encoder, imnet_q, imnet_k, imnet_v)
    model1.load_state_dict(torch.load('rdn.pth'))
    model1.eval()
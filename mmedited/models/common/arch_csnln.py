from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg

from mmedited.models.common.vgg_arch import VGGFeatureExtractor
import pdb


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks



class ContrasExtractorLayer(nn.Module):

    def __init__(self):
        super(ContrasExtractorLayer, self).__init__()

        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
            'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv3_1_idx = vgg16_layers.index('conv3_1')
        features = getattr(vgg, 'vgg16')(pretrained=True).features[:conv3_1_idx + 1]
        modified_net = OrderedDict()
        for k, v in zip(vgg16_layers, features):
            modified_net[k] = v

        self.model = nn.Sequential(modified_net)
        # the mean is for image with range [0, 1]
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, batch):
        batch = (batch - self.mean) / self.std
        output = self.model(batch)
        return output


class ContrasExtractorSep(nn.Module):

    def __init__(self):
        super(ContrasExtractorSep, self).__init__()

        self.feature_extraction_image1 = ContrasExtractorLayer()
        self.feature_extraction_image2 = ContrasExtractorLayer()

    def forward(self, image1, image2):
        dense_features1 = self.feature_extraction_image1(image1)
        dense_features2 = self.feature_extraction_image2(image2)

        return {
            'dense_features1': dense_features1,
            'dense_features2': dense_features2
        }


def sample_patches(inputs, patch_size=3, stride=1):
    """Extract sliding local patches from an input feature tensor.
    The sampled pathes are row-major.
    Args:
        inputs (Tensor): the input feature maps, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
    Returns:
        patches (Tensor): extracted patches, shape: (c, patch_size,
            patch_size, n_patches).
    """

    c, h, w = inputs.shape
    patches = inputs.unfold(1, patch_size, stride)\
                    .unfold(2, patch_size, stride)\
                    .reshape(c, -1, patch_size, patch_size)\
                    .permute(0, 2, 3, 1)
    return patches


def feature_match_index(feat_input,
                        feat_ref,
                        patch_size=3,
                        input_stride=1,
                        ref_stride=1,
                        is_norm=True,
                        norm_input=False):
    """Patch matching between input and reference features.
    Args:
        feat_input (Tensor): the feature of input, shape: (c, h, w).
        feat_ref (Tensor): the feature of reference, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
        is_norm (bool): determine to normalize the ref feature or not.
            Default:True.
    Returns:
        max_idx (Tensor): The indices of the most similar patches.
        max_val (Tensor): The correlation values of the most similar patches.
    """

    # patch decomposition, shape: (c, patch_size, patch_size, n_patches)
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)

    # normalize reference feature for each patch in both channel and
    # spatial dimensions.

    # batch-wise matching because of memory limitation
    _, h, w = feat_input.shape
    batch_size = int(1024.**2 * 512 / (h * w))
    n_patches = patches_ref.shape[-1]

    max_idx, max_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)
        corr = F.conv2d(
            feat_input.unsqueeze(0),
            batch.permute(3, 0, 1, 2),
            stride=input_stride)

        max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0)

        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp
        else:
            indices = max_val_tmp > max_val
            max_val[indices] = max_val_tmp[indices]
            max_idx[indices] = max_idx_tmp[indices] + idx

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(
            int((h - patch_size) / input_stride + 1),
            int((w - patch_size) / input_stride + 1))
        max_val = max_val / norm

    return max_idx, max_val


def tensor_shift(x, shift=(2, 2), fill_val=0):
    """ Tensor shift.
    Args:
        x (Tensor): the input tensor. The shape is [b, h, w, c].
        shift (tuple): shift pixel.
        fill_val (float): fill value
    Returns:
        Tensor: the shifted tensor.
    """

    _, h, w, _ = x.size()
    shift_h, shift_w = shift
    new = torch.ones_like(x) * fill_val

    if shift_h >= 0 and shift_w >= 0:
        len_h = h - shift_h
        len_w = w - shift_w
        new[:, shift_h:shift_h + len_h,
            shift_w:shift_w + len_w, :] = x.narrow(1, 0,
                                                   len_h).narrow(2, 0, len_w)
    else:
        raise NotImplementedError
    return new


class CorrespondenceFeatGenerationArch(nn.Module):
    def __init__(self,
                 patch_size=3,
                 stride=1,
                 vgg_layer_list=['relu3_1'],
                 vgg_type='vgg19'):
        super(CorrespondenceFeatGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.vgg_layer_list = vgg_layer_list
        self.vgg = VGGFeatureExtractor(
            layer_name_list=vgg_layer_list, vgg_type=vgg_type)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                           dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid  # shape:(1, w, h, 2)
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def forward(self, dense_features, img_ref_hr):
        batch_offset_relu = []

        for ind in range(img_ref_hr.size(0)):
            feat_in = dense_features['dense_features1'][ind]
            feat_ref = dense_features['dense_features2'][ind]
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(feat_ref.reshape(c, -1), dim=0).view(c, h, w)

            _max_idx, _max_val = feature_match_index(
                feat_in,
                feat_ref,
                patch_size=self.patch_size,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)

            # offset map for relu3_1
            offset_relu3 = self.index_to_flow(_max_idx)
            # shift offset relu3
            shifted_offset_relu3 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu3, (i, j))
                    shifted_offset_relu3.append(flow_shift)
            shifted_offset_relu3 = torch.cat(shifted_offset_relu3, dim=0)
            batch_offset_relu.append(shifted_offset_relu3)

        # size: [b, 9, h, w, 2], the order of the last dim: [x, y]
        batch_offset_relu = torch.stack(batch_offset_relu, dim=0)

        img_ref_feat = self.vgg(img_ref_hr)
        return batch_offset_relu, img_ref_feat


class CorrespondenceGenerationArch(nn.Module):
    def __init__(self,
                 patch_size=3,
                 stride=1):
        super(CorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

    def index_to_flow(self, max_idx):

        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).to(device), torch.arange(0, w).to(device))
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float()  #to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                           dim=2).unsqueeze(0).float()  #to(device)
        flow = flow - grid  # shape:(1, w, h, 2)
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2)).cuda()

        return flow

    def forward(self, feats_in, feats_ref):
        batch_offset_relu = []

        for ind in range(feats_in.size(0)):
            feat_in = feats_in[ind]
            feat_ref = feats_ref[ind]

            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(feat_ref.reshape(c, -1), dim=0).view(c, h//2, w//2)

            _max_idx, _max_val = feature_match_index(
                feat_in,
                feat_ref,
                patch_size=self.patch_size,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)

            # offset map
            offset = self.index_to_flow(_max_idx)
            # shift offset
            shifted_offset = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset, (i, j))
                    shifted_offset.append(flow_shift)
            shifted_offset = torch.cat(shifted_offset, dim=0)
            batch_offset_relu.append(shifted_offset)

        # size: [b, 9, h, w, 2], the order of the last dim: [x, y]
        batch_offset_relu = torch.stack(batch_offset_relu, dim=0)

        return batch_offset_relu


class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat


#cross-scale non-local attention
class CrossScaleAttention(nn.Module):
    def __init__(self, channel=64, reduction=2, ksize=3, scale=2, stride=1, softmax_scale=10, average=True, conv=default_conv):
        super(CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale

        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())  
        self.conv_match_2 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())  
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())    
        #self.register_buffer('fuse_weight', fuse_weight)

        if 3 in scale:
            self.downx3 = nn.Conv2d(channel, channel, ksize, 3, 1)
        if 4 in scale:
            self.downx4 = nn.Conv2d(channel, channel, ksize, 4, 1)

        self.down = nn.Conv2d(channel, channel, ksize, 2, 1)    

    def forward(self, input):
        _, _, H, W = input.shape

        if not isinstance(self.scale, list):
            self.scale = [self.scale]

        res_y = []
        for s in self.scale:
            
            # if (H%2 != 0):
            #     input = F.pad(input, (0, 0, 0, 1), "constant", 0)
            # if (W%2 != 0):
            #     input = F.pad(input, (0, 1, 0, 0), "constant", 0)

            mod_pad_h, mod_pad_w = 0, 0
            if H % s != 0:
                mod_pad_h = s - H % s
            if W % s != 0:
                mod_pad_w = s - W % s
            input_pad = F.pad(input, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            #get embedding
            embed_w = self.conv_assembly(input_pad)     # [16, 64, 48, 48]
            match_input = self.conv_match_1(input_pad)  # [16, 32, 48, 48]

            # b*c*h*w
            shape_input = list(embed_w.size())      # b*c*h*w
            input_groups = torch.split(match_input, 1, dim=0)  # 16x[1, 32, 48, 48]
            # kernel size on input for matching
            kernel = s * self.ksize

            # raw_w is extracted for reconstruction
            raw_w = extract_image_patches(embed_w, ksizes=[kernel, kernel],
                                        strides=[self.stride * s, self.stride * s],
                                        rates=[1, 1],
                                        padding='same') # [16, 2304, 576], 2304=64*6*6, 576=48*48/(2*2), [N, C*k*k, L] 

            # raw_shape: [N, C, k, k, L]
            raw_w = raw_w.view(shape_input[0], shape_input[1], kernel, kernel, -1) # [16, 64, 6, 6, 576]
            raw_w = raw_w.permute(0, 4, 1, 2, 3).contiguous()    # [16, 576, 64, 6, 6] raw_shape: [N, L, C, k, k]
            raw_w_groups = torch.split(raw_w, 1, dim=0)  # 16x[1, 576, 64, 6, 6]


            # downscaling X to form Y for cross-scale matching
            ref = F.interpolate(input_pad, scale_factor=1./s, mode='bilinear')  # [16, 64, 24, 24]
            ref = self.conv_match_2(ref)        # [16, 32, 24, 24]
            w = extract_image_patches(ref, ksizes=[self.ksize, self.ksize],
                                    strides=[self.stride, self.stride],
                                    rates=[1, 1],
                                    padding='same')   # [16, 288, 576], 288=32*3*3, 576=24*24
            shape_ref = ref.shape
            
            # w shape: [N, C, k, k, L]
            w = w.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1) # [16, 32, 3, 3, 576]
            w = w.permute(0, 4, 1, 2, 3).contiguous()    # [16, 576, 32, 3, 3] w shape: [N, L, C, k, k]
            w_groups = torch.split(w, 1, dim=0)     # 16x[1, 576, 32, 3, 3]

            y = []
            # 1*1*k*k
            #fuse_weight = self.fuse_weight

            for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
                # normalize
                wi = wi[0]  # [576, 32, 3, 3] [L, C, k, k]
                max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                    axis=[1, 2, 3], keepdim=True)), self.escape_NaN) # 
                wi_normed = wi/ max_wi # 
                
                # Compute correlation map
                xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # [1, 32, 50, 50]  xi: 1*c*H*W
                yi = F.conv2d(xi, wi_normed, stride=1)   # [1, 576, 48, 48] [1, L, H, W] L = shape_ref[2]*shape_ref[3]
                # yi = F.conv2d(xi.cpu(), wi_normed.cpu(), stride=1)  #TODO

                yi = yi.view(1, shape_ref[2] * shape_ref[3], shape_input[2], shape_input[3])  # [1, 576, 48, 48]  (B=1, C=32*32, H=32, W=32)
                # rescale matching score
                yi = F.softmax(yi*self.softmax_scale, dim=1)     # [1, 576, 48, 48]
                if self.average == False:
                    yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()

                # deconv for reconsturction
                wi_center = raw_wi[0]   # [576, 64, 6, 6]
                yi = F.conv_transpose2d(yi, wi_center, stride=self.stride*s, padding=s)   #[1, 64, 96, 96]
                # yi = F.conv_transpose2d(yi, wi_center.cpu(), stride=self.stride*s, padding=s).cuda()  #TODO

                # add down
                if s == 2:
                    yi = self.down(yi)  #[1, 64, 48, 48]
                elif s == 3:
                    yi = self.downx3(yi)
                elif s == 4:
                    yi = self.downx4(yi)

                yi =yi/6.
                y.append(yi)

            y = torch.cat(y, dim=0)
            y = y[:, :, :H, :W]

            res_y.append(y)
        
        res_y = torch.cat(res_y, dim=1)

        return res_y  #y
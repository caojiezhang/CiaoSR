import io
import logging
import random
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scipy.linalg import orth

from mmedit.datasets.pipelines import blur_kernels as blur_kernels

from mmedited.utils import utils_image as util

try:
    import av
    has_av = True
except ImportError:
    has_av = False
import pdb


# @PIPELINES.register_module()
class RandomBlur:
    """Apply random blur to the input.
    Modified keys are the attributed specified in "keys".
    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def get_kernel(self, num_kernels):
        kernel_type = np.random.choice(
            self.params['kernel_list'], p=self.params['kernel_prob'])
        kernel_size = random.choice(self.params['kernel_size'])

        sigma_x_range = self.params.get('sigma_x', [0, 0])
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_x_step = self.params.get('sigma_x_step', 0)

        sigma_y_range = self.params.get('sigma_y', [0, 0])
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        sigma_y_step = self.params.get('sigma_y_step', 0)

        rotate_angle_range = self.params.get('rotate_angle', [-np.pi, np.pi])
        rotate_angle = np.random.uniform(rotate_angle_range[0],
                                         rotate_angle_range[1])
        rotate_angle_step = self.params.get('rotate_angle_step', 0)

        beta_gau_range = self.params.get('beta_gaussian', [0.5, 4])
        beta_gau = np.random.uniform(beta_gau_range[0], beta_gau_range[1])
        beta_gau_step = self.params.get('beta_gaussian_step', 0)

        beta_pla_range = self.params.get('beta_plateau', [1, 2])
        beta_pla = np.random.uniform(beta_pla_range[0], beta_pla_range[1])
        beta_pla_step = self.params.get('beta_plateau_step', 0)

        omega_range = self.params.get('omega', None)
        omega_step = self.params.get('omega_step', 0)
        if omega_range is None:  # follow Real-ESRGAN settings if not specified
            if kernel_size < 13:
                omega_range = [np.pi / 3., np.pi]
            else:
                omega_range = [np.pi / 5., np.pi]
        omega = np.random.uniform(omega_range[0], omega_range[1])

        # determine blurring kernel
        kernels = []
        for _ in range(0, num_kernels):
            kernel = blur_kernels.random_mixed_kernels(
                [kernel_type],
                [1],
                kernel_size,
                [sigma_x, sigma_x],
                [sigma_y, sigma_y],
                [rotate_angle, rotate_angle],
                [beta_gau, beta_gau],
                [beta_pla, beta_pla],
                [omega, omega],
                None,
            )
            kernels.append(kernel)

            # update kernel parameters
            sigma_x += np.random.uniform(-sigma_x_step, sigma_x_step)
            sigma_y += np.random.uniform(-sigma_y_step, sigma_y_step)
            rotate_angle += np.random.uniform(-rotate_angle_step,
                                              rotate_angle_step)
            beta_gau += np.random.uniform(-beta_gau_step, beta_gau_step)
            beta_pla += np.random.uniform(-beta_pla_step, beta_pla_step)
            omega += np.random.uniform(-omega_step, omega_step)

            sigma_x = np.clip(sigma_x, sigma_x_range[0], sigma_x_range[1])
            sigma_y = np.clip(sigma_y, sigma_y_range[0], sigma_y_range[1])
            rotate_angle = np.clip(rotate_angle, rotate_angle_range[0],
                                   rotate_angle_range[1])
            beta_gau = np.clip(beta_gau, beta_gau_range[0], beta_gau_range[1])
            beta_pla = np.clip(beta_pla, beta_pla_range[0], beta_pla_range[1])
            omega = np.clip(omega, omega_range[0], omega_range[1])

        return kernels

    def _apply_random_blur(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # get kernel and blur the input
        kernels = self.get_kernel(num_kernels=len(imgs))
        imgs = [
            cv2.filter2D(img, -1, kernel)
            for img, kernel in zip(imgs, kernels)
        ]

        if is_single_image:
            imgs = imgs[0]

        return imgs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_blur(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str

class RandomJPEGCompression:
    """Apply random JPEG compression to the input.
    Modified keys are the attributed specified in "keys".
    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_random_compression(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # determine initial compression level and the step size
        quality = self.params['quality']
        quality_step = self.params.get('quality_step', 0)
        jpeg_param = round(np.random.uniform(quality[0], quality[1]))

        # apply jpeg compression
        outputs = []
        for img in imgs:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_param]
            _, img_encoded = cv2.imencode('.jpg', img * 255., encode_param)
            outputs.append(np.float32(cv2.imdecode(img_encoded, 1)) / 255.)

            # update compression level
            jpeg_param += np.random.uniform(-quality_step, quality_step)
            jpeg_param = round(np.clip(jpeg_param, quality[0], quality[1]))

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_compression(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class UVSRDegradation:
    """Apply uvsr degradation to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def downsample1D(self, x, sf=2):
        '''s-fold downsampler

        Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

        x: tensor image, NxCxWxH
        '''
        st = 0
        return x[st::sf, ...]

    def _apply_uvsr_degradation(self, imgs):
        # get kernel and blur the input

        T, C, H, W = imgs.shape             #[100, 3, 256, 256]
        N = self.params['fuse_frames']

        assert N == 5
        assert T > N
        t = T // N
        out_imgs = []

        padding = 'reflection'
        if padding == 'reflection':
            imgs = torch.cat((imgs[1:2, ...], imgs[0:1, ...], imgs, imgs[-2:-1, ...], imgs[-3:-2, ...]), dim=0)
            for i in range(len(imgs)):
                if i>=2 and i+3<=len(imgs):
                    out_imgs.append(imgs[i-2:i+3, ...].mean(0, keepdim=True))
        else:
            for i in range(t):
                out_imgs.append(imgs[i*N: i*N+N, ...].mean(0, keepdim=True))
        # # imgs = imgs.mean(0, keepdim=True)

        out_imgs = torch.cat(out_imgs, dim=0)

        # F.conv3d

        # downsampling
        if self.params['down']:
            sf = self.params['scale']
            out_imgs = self.downsample1D(out_imgs, sf=sf[0])
            # out_imgs = F.interpolate(out_imgs.view(-1, C, H, W), size=(H//sf[1], W//sf[2]), mode='bicubic')

            assert sf[1] == sf[2]
            down_imgs = []
            for i in range(out_imgs.shape[0]):
                down_img = util.imresize(out_imgs[i].permute(1,2,0), 1./sf[1])
                down_imgs.append(down_img)
            out_imgs = torch.stack(down_imgs, dim=0).permute(0,3,1,2)
        # noise

        return out_imgs #imgs

    def __call__(self, results):
        for key in self.keys:
            results[key] = self._apply_uvsr_degradation(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RandomVideoCompression:
    """Apply random video compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        assert has_av, 'Please install av to use video compression.'

        self.keys = keys
        self.params = params
        logging.getLogger('libav').setLevel(50)

    def _apply_random_compression(self, imgs):
        codec = random.choices(self.params['codec'],
                               self.params['codec_prob'])[0]
        bitrate = self.params['bitrate']
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = imgs[0].shape[0]
            stream.width = imgs[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = bitrate

            for img in imgs:
                img = np.uint8((img.clip(0, 1) * 255.).round()) # fix error in img = (255 * img).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame.pict_type = 'NONE'
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(
                        frame.to_rgb().to_ndarray().astype(np.float32) / 255.)

        return outputs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_compression(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RandomGaussianNoise:
    """Apply random Gaussian to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_gaussian_noise(self, imgs):

        sigma_range = self.params['gaussian_sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * sigma
            if self.params['nonblind_noise']:
                outputs.append(np.concatenate((img+noise, np.broadcast_to(sigma, (img.shape[0], img.shape[1], 1))), axis=2))
            else:
                outputs.append(img+noise)

        return outputs

    def __call__(self, results):
        # if np.random.uniform() > self.params.get('prob', 1):
        #     return results

        for key in self.keys:
            results[key] = self._apply_gaussian_noise(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RandomGaussianNoisewithPro:
    """Apply random Gaussian to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_gaussian_noise(self, imgs):

        if random.random() < self.params['prob']:
            level_range = self.params['gaussian_beta']
        else:
            level_range = self.params['gaussian_sigma']
        level = np.random.uniform(level_range[0], level_range[1]) / 255.

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * level
            if self.params['nonblind_noise']:
                outputs.append(np.concatenate((img+noise, np.broadcast_to(level, (img.shape[0], img.shape[1], 1))), axis=2))
            else:
                outputs.append(img+noise)

        return outputs

    def __call__(self, results):
        # if np.random.uniform() > self.params.get('prob', 1):
        #     return results

        for key in self.keys:
            results[key] = self._apply_gaussian_noise(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RandomClipGaussianNoise:
    """Apply random Gaussian to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_gaussian_noise(self, imgs):

        sigma_range = self.params['gaussian_sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * sigma
            # noise = np.clip(noise, 0, 1)
            if self.params['nonblind_noise']:
                outputs.append(np.concatenate((np.clip(img+noise, 0, 1), np.broadcast_to(sigma, (img.shape[0], img.shape[1], 1))), axis=2))
            else:
                outputs.append(np.clip(img+noise, 0, 1))

        return outputs

    def __call__(self, results):
        # if np.random.uniform() > self.params.get('prob', 1):
        #     return results

        for key in self.keys:
            results[key] = self._apply_gaussian_noise(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')


class RandomLevel:
    """Concat random level to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_level(self, imgs):

        sigma_range = self.params['sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.

        outputs = []
        for img in imgs:
            outputs.append(np.concatenate((img, np.broadcast_to(sigma, (img.shape[0], img.shape[1], 1))), axis=2))

        return outputs

    def __call__(self, results):
        # if np.random.uniform() > self.params.get('prob', 1):
        #     return results

        for key in self.keys:
            results[key] = self._apply_level(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str

class OldRandomNoise:
    """Apply random real noise to the input.

    Currently support Gaussian noise and Poisson noise.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        assert has_av, 'Please install av to use video compression.'

        self.keys = keys
        self.params = params
        logging.getLogger('libav').setLevel(50)

    def _add_blur(self, imgs, imgs_hq):
        if random.random() < self.params['blur_prob']:
            sf = random.choice(self.params['sf'])   #[2, 3, 4]
            wd = 2.0 + 0.2*sf
            k = fspecial('gaussian', 2*random.randint(2, 11)+3, wd*random.random())

            outputs, outputs_hq = [], []
            for img, hq in zip(imgs, imgs_hq):
                img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
                hq = ndimage.filters.convolve(hq, np.expand_dims(k, axis=2), mode='mirror')
                outputs.append(img)
                outputs_hq.append(hq)
            # print("blur", imgs.shape)
        return outputs, outputs_hq

    def _add_resize(self, imgs, imgs_hq):
        inter_method = random.choice(self.params['inter_method'])  #[1, 2, 3]

        rnum = np.random.rand()
        if rnum > 0.75:    # up
            sf = random.uniform(1, 2)
        elif rnum < 0.25:  # down
            sf = random.uniform(0.5, 1)
        else:
            sf = 1.0

        if sf != 1.0:
            outputs, outputs_hq = [], []
            for img, hq in zip(imgs, imgs_hq):
                img_W0, img_H0 = img.shape[1], img.shape[0]
                img_W, img_H = int(sf*img_W0)//2*2, int(sf*img_H0)//2*2
                img = cv2.resize(img, (img_W, img_H), interpolation=inter_method)
                img = cv2.resize(img, (img_W0, img_H0), interpolation=inter_method)
                img = np.clip(img, 0.0, 1.0)

                hq_W0, hq_H0 = hq.shape[1], hq.shape[0]
                hq_W, hq_H = int(sf*hq_W0)//2*2, int(sf*hq_H0)//2*2
                hq = cv2.resize(hq, (hq_W, hq_H), interpolation=inter_method)
                hq = cv2.resize(hq, (hq_W0, hq_H0), interpolation=inter_method)
                hq = np.clip(hq, 0.0, 1.0)

                outputs.append(img)
                outputs_hq.append(hq)
            # print("Resizing", imgs.shape)
        return img, hq

    def _add_Gaussian_noise(self, imgs):

        sigma_range = self.params['gaussian_sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.

        sigma_step = self.params.get('gaussian_sigma_step', 0)

        gray_noise_prob = self.params['gaussian_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * sigma
            if is_gray_noise:
                noise = noise[:, :, :1]
            noisy_img = img + noise
            noisy_img = np.clip(noisy_img, 0.0, 1.0)
            outputs.append(noisy_img)

            # update noise level
            sigma += np.random.uniform(-sigma_step, sigma_step) / 255.
            sigma = np.clip(sigma, sigma_range[0] / 255.,
                            sigma_range[1] / 255.)

        return outputs  #np.stack(outputs, axis=0)

    def _add_speckle_noise(self, imgs):
        noise_levels = self.params['speckle_level']  #[10, 50]

        outputs = []
        for img in imgs:
            noise_level = random.randint(noise_levels[0], noise_levels[1])
            img = np.clip(img, 0.0, 1.0)
            rnum = random.random()
            if rnum > 0.6:
                img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
            elif rnum < 0.4:
                img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
            else:
                L = noise_levels[1]/255.
                D = np.diag(np.random.rand(3))
                U = orth(np.random.rand(3,3))
                conv = np.dot(np.dot(np.transpose(U), D), U)
                img += img*np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
            outputs.append(img)

        return outputs

    def _add_Poisson_noise(self, imgs):
        scale_range = self.params['poisson_scale']
        scale = np.random.uniform(scale_range[0], scale_range[1])

        scale_step = self.params.get('poisson_scale_step', 0)

        gray_noise_prob = self.params['poisson_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = img.copy()
            if is_gray_noise:
                noise = cv2.cvtColor(noise[..., [2, 1, 0]], cv2.COLOR_BGR2GRAY)
                noise = noise[..., np.newaxis]
            noise = np.clip((noise * 255.0).round(), 0, 255) / 255.
            unique_val = 2**np.ceil(np.log2(len(np.unique(noise))))
            noise = np.random.poisson(noise * unique_val) / unique_val - noise

            outputs.append(img + noise * scale)

            # update noise level
            scale += np.random.uniform(-scale_step, scale_step)
            scale = np.clip(scale, scale_range[0], scale_range[1])

        return outputs

    def _add_JPEG_noise(self, imgs):
        quality_range = self.params['quality_range']  #(20, 95)

        outputs = []
        for img in imgs:
            quality_factor = random.randint(quality_range[0], quality_range[1])
            img = cv2.cvtColor(np.uint8((img.clip(0, 1)*255.).round()), cv2.COLOR_RGB2BGR)
            result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img = cv2.imdecode(encimg, 1)
            img = cv2.cvtColor(np.float32(img/255.), cv2.COLOR_BGR2RGB)
            outputs.append(img)
        return outputs


    def _add_video_compression(self, imgs):
        codec = random.choices(self.params['codec'], self.params['codec_prob'])[0]
        bitrate = self.params['bitrate']
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = imgs[0].shape[0]
            stream.width = imgs[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = bitrate

            for img in imgs:
                img = (255 * img).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame.pict_type = 'NONE'
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(
                        frame.to_rgb().to_ndarray().astype(np.float32) / 255.)

        # print("Video compression", imgs.shape)

        return outputs

    def _add_shift(self, imgs, imgs_hq):
        if random.random() < self.params['shift_prob']:
            outputs, outputs_hq = [], []
            for img, hq in zip(imgs, imgs_hq):
                # h_shift, w_shift = random.randint(0,8), random.randint(0,8)
                h_shift, w_shift = 2*random.randint(0,4), 2*random.randint(0,4)  # must be even due to video compression
                imgs, imgs_hq = img[h_shift:, w_shift:, ...], hq[h_shift:, w_shift:, ...]
                outputs.append(img)
                outputs_hq.append(hq)
        return outputs, outputs_hq

    def _add_real_random_noise(self, imgs):

        # if not isinstance(imgs, np.ndarray):
        #     imgs = np.array(imgs)
        imgs, imgs_hq = imgs.copy(), imgs.copy()

        n_deg = 16
        if random.random() < self.params['shuffle_prob']:
            shuffle_order = random.sample(range(n_deg), n_deg)
        else:
            shuffle_order = list(range(n_deg))

        # local shuffle for noise, JPEG is always the last one, we set the video compression at the last
        # shuffle_order[2:6] = random.sample(shuffle_order[2:6], len(range(2, 6)))
        # shuffle_order[9:13] = random.sample(shuffle_order[9:13], len(range(9, 13)))

        for i in shuffle_order:
            # blur
            if i == 0:
                imgs, imgs_hq = self._add_blur(imgs, imgs_hq)
            # Resizing
            elif i == 1:
                imgs, imgs_hq = self._add_resize(imgs, imgs_hq)
            # Gaussian noise
            elif i == 2:
                imgs = self._add_Gaussian_noise(imgs)
            # Poisson noise
            elif i == 3:
                imgs = self._add_Poisson_noise(imgs)
            # Speckle noise
            elif i == 4:
                imgs = self._add_speckle_noise(imgs)
            # ISP
            # elif i == 5:
            #     imgs = self._add_isp_noise(imgs)
            # JPEG noise
            elif i == 6:
                imgs = self._add_JPEG_noise(imgs)
            # Video compression
            elif i == 7:
                imgs = self._add_video_compression(imgs)
            # Resizing
            elif i == 8:
                imgs, imgs_hq = self._add_resize(imgs, imgs_hq)
            # Gaussian noise
            elif i == 9:
                imgs = self._add_Gaussian_noise(imgs)
            # Poisson noise
            elif i == 10:
                imgs = self._add_Poisson_noise(imgs)
            # Speckle noise
            elif i == 11:
                imgs = self._add_speckle_noise(imgs)
            # ISP
            # elif i == 12:
            #     imgs = self._add_isp_noise(imgs)
            # Random shift
            elif i == 13:
                imgs, imgs_hq = self._add_shift(imgs, imgs_hq)
            # JPEG noise
            elif i == 14:
                imgs = self._add_JPEG_noise(imgs)
            # Video compression
            elif i == 15:
                imgs = self._add_video_compression(imgs)

            # imgs = imgs.transpose(0,3,1,2)
            # imgs_hq = imgs_hq.transpose(0,3,1,2)

        # imgs = list(imgs)
        # imgs_hq = list(imgs_hq)

        return imgs, imgs_hq

    def __call__(self, results):
        # if np.random.uniform() > self.params.get('prob', 1):
        #     return results
        results['lq'], results['gt'] = self._add_real_random_noise(results['gt'])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RealRandomNoise:
    """Apply random noise to the input.

    Currently support Gaussian noise and Poisson noise.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_gaussian_noise(self, imgs):
        sigma_range = self.params['gaussian_sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.

        sigma_step = self.params.get('gaussian_sigma_step', 0)

        gray_noise_prob = self.params['gaussian_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * sigma
            if is_gray_noise:
                noise = noise[:, :, :1]
            outputs.append(img + noise)

            # update noise level
            sigma += np.random.uniform(-sigma_step, sigma_step) / 255.
            sigma = np.clip(sigma, sigma_range[0] / 255.,
                            sigma_range[1] / 255.)

        return outputs

    def _apply_poisson_noise(self, imgs):
        scale_range = self.params['poisson_scale']
        scale = np.random.uniform(scale_range[0], scale_range[1])

        scale_step = self.params.get('poisson_scale_step', 0)

        gray_noise_prob = self.params['poisson_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = img.copy()
            if is_gray_noise:
                noise = cv2.cvtColor(noise[..., [2, 1, 0]], cv2.COLOR_BGR2GRAY)
                noise = noise[..., np.newaxis]
            noise = np.clip((noise * 255.0).round(), 0, 255) / 255.
            unique_val = 2**np.ceil(np.log2(len(np.unique(noise))))
            noise = np.random.poisson(noise * unique_val) / unique_val - noise

            outputs.append(img + noise * scale)

            # update noise level
            scale += np.random.uniform(-scale_step, scale_step)
            scale = np.clip(scale, scale_range[0], scale_range[1])

        return outputs

    def _apply_speckle_noise(self, imgs):
        noise_levels = self.params['speckle_level']

        outputs = []
        for img in imgs:
            noise_level = random.randint(noise_levels[0], noise_levels[1])
            img = np.clip(img, 0.0, 1.0)
            rnum = random.random()
            if rnum > 0.6:
                img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
            elif rnum < 0.4:
                img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
            else:
                L = noise_levels[1]/255.
                D = np.diag(np.random.rand(3))
                U = orth(np.random.rand(3,3))
                conv = np.dot(np.dot(np.transpose(U), D), U)
                img += img*np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
            outputs.append(img)
        return outputs

    def _apply_jpeg_noise(self, imgs):
        quality_range = self.params['quality_range']

        outputs = []
        for img in imgs:
            quality_factor = random.randint(quality_range[0], quality_range[1])
            img = cv2.cvtColor(np.uint8((img.clip(0, 1)*255.).round()), cv2.COLOR_RGB2BGR)
            result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img = cv2.imdecode(encimg, 1)
            img = cv2.cvtColor(np.float32(img/255.), cv2.COLOR_BGR2RGB)
            outputs.append(img)
        return outputs


    def _apply_random_noise(self, imgs):

        noise_type = self.params['noise_type']
        n_noise = len(noise_type)
        if random.random() < self.params['shuffle_prob']:
            shuffle_order = random.sample(range(n_noise), n_noise)
        else:
            shuffle_order = list(range(n_noise))

        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        for i in shuffle_order:
            if noise_type[i] == 'gaussian':
                imgs = self._apply_gaussian_noise(imgs)
            elif noise_type[i] == 'poisson':
                if random.random() < self.params['other_prob']:
                    imgs = self._apply_poisson_noise(imgs)
            elif noise_type[i] == 'speckle':
                if random.random() < self.params['other_prob']:
                    imgs = self._apply_speckle_noise(imgs)
            elif noise_type[i] == 'isp':
                if random.random() < self.params['other_prob']:
                    imgs = self._apply_isp_noise(imgs)
            elif noise_type[i] == 'jpeg':
                imgs = self._apply_jpeg_noise(imgs)
            else:
                raise NotImplementedError(f'"noise_type" [{noise_type}] is '
                                        'not implemented.')

        if is_single_image:
            imgs = imgs[0]

        return imgs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_noise(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RealRandomResize:
    """Randomly resize the input.
    Modified keys are the attributed specified in "keys".
    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

        self.resize_dict = dict(
            bilinear=cv2.INTER_LINEAR,
            bicubic=cv2.INTER_CUBIC,
            area=cv2.INTER_AREA,
            lanczos=cv2.INTER_LANCZOS4)

    def _random_resize(self, imgs, resize_opt, scale_factor):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        h, w = imgs[0].shape[:2]
        ori_size = (h, w)

        # determine output size
        h_out, w_out = h * scale_factor, w * scale_factor
        if self.params.get('is_size_even', False):
            h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
        target_size = (int(h_out), int(w_out))

        # resize all input with the same target_size
        outputs = [
            cv2.resize(
                cv2.resize(img, target_size[::-1], interpolation=resize_opt),
                ori_size[::-1], interpolation=resize_opt)
            for img in imgs
        ]

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        resize_opt = self.params['resize_opt']
        resize_prob = self.params['resize_prob']
        resize_opt = np.random.choice(resize_opt, p=resize_prob).lower()
        if resize_opt not in self.resize_dict:
            raise NotImplementedError(f'resize_opt [{resize_opt}] is not '
                                      'implemented')
        resize_opt = self.resize_dict[resize_opt]

        resize_mode = np.random.choice(['up', 'down', 'keep'],
                                           p=self.params['resize_mode_prob'])

        resize_scale = self.params['resize_scale']
        if resize_mode == 'up':
            scale_factor = np.random.uniform(1, resize_scale[1])
        elif resize_mode == 'down':
            scale_factor = np.random.uniform(resize_scale[0], 1)
        else:
            scale_factor = 1

        for key in self.keys:
            results[key] = self._random_resize(results[key], resize_opt, scale_factor)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class NorResize:
    """Randomly resize the input.
    Modified keys are the attributed specified in "keys".
    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

        self.resize_dict = dict(
            bilinear=cv2.INTER_LINEAR,
            bicubic=cv2.INTER_CUBIC,
            area=cv2.INTER_AREA,
            lanczos=cv2.INTER_LANCZOS4)

    def _random_resize(self, imgs_list):
        imgs = imgs_list['gt']
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        resize_opt = self.params['resize_opt']
        # resize_prob = self.params['resize_prob']
        # resize_opt = np.random.choice(resize_opt, p=resize_prob).lower()
        if resize_opt not in self.resize_dict:
            raise NotImplementedError(f'resize_opt [{resize_opt}] is not '
                                      'implemented')
        resize_opt = self.resize_dict[resize_opt]

        resize_step = self.params.get('resize_step', 0)

        # determine the target size, if not provided
        target_size = self.params.get('target_size', None)
        resize_scale = self.params['resize_scale']
        scale_factor = np.random.uniform(resize_scale[0], resize_scale[1])

        w_hr = round(target_size * scale_factor)
        x0 = np.random.randint(0, img.shape[-3] - w_hr)
        y0 = np.random.randint(0, img.shape[-2] - w_hr)
        crop_lr_pre = img[x0:x0 + w_hr, y0:y0 + w_hr, :]

        if target_size is None:
            resize_mode = np.random.choice(['up', 'down', 'keep'],
                                           p=self.params['resize_mode_prob'])
            resize_scale = self.params['resize_scale']
            if resize_mode == 'up':
                scale_factor = np.random.uniform(1, resize_scale[1])
            elif resize_mode == 'down':
                scale_factor = np.random.uniform(resize_scale[0], 1)
            else:
                scale_factor = 1

            patch_size = self.params['patch_size']
            if patch_size is None:
                h, w = imgs[0].shape[:2]
            else:
                h = w = self.patch_size

            # determine output size
            h_out, w_out = h * scale_factor, w * scale_factor
            if self.params.get('is_size_even', False):
                h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
            target_size = (int(h_out), int(w_out))
        else:
            resize_step = 0

        # resize the input
        if resize_step == 0:  # same target_size for all input images
            outputs = [
                cv2.resize(img, target_size[::-1], interpolation=resize_opt)
                for img in imgs
            ]
        else:  # different target_size for each input image
            outputs = []
            for img in imgs:
                img = cv2.resize(
                    img, target_size[::-1], interpolation=resize_opt)
                outputs.append(img)

                # update scale
                scale_factor += np.random.uniform(-resize_step, resize_step)
                scale_factor = np.clip(scale_factor, resize_scale[0],
                                       resize_scale[1])

                # determine output size
                h_out, w_out = h * scale_factor, w * scale_factor
                if self.params.get('is_size_even', False):
                    h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
                target_size = (int(h_out), int(w_out))

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, results):
        results = self._random_resize(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


class RandomScaleResize1:
    """Randomly resize the input with random scale.
    Modified keys are the attributed specified in "keys".
    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

        self.resize_dict = dict(
            bilinear=cv2.INTER_LINEAR,
            bicubic=cv2.INTER_CUBIC,
            area=cv2.INTER_AREA,
            lanczos=cv2.INTER_LANCZOS4)


    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        imgs = results['lq']
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        h, w = results['gt'].shape[:2]

        resize_opt = self.params['resize_opt']
        resize_prob = self.params['resize_prob']
        resize_opt = np.random.choice(resize_opt, p=resize_prob).lower()
        if resize_opt not in self.resize_dict:
            raise NotImplementedError(f'resize_opt [{resize_opt}] is not '
                                      'implemented')
        resize_opt = self.resize_dict[resize_opt]

        # determine the target size, if not provided
        target_size = self.params.get('target_size', None)
        if target_size is None:
            scale_min = self.params['scale_min']
            scale_max = self.params['scale_max']
            scale_factor = 1./np.random.uniform(scale_min, scale_max)

            # determine output size
            h_out, w_out = h * scale_factor, w * scale_factor
            if self.params.get('is_size_even', False):
                h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
            target_size = (int(h_out), int(w_out))
        
        # resize the input
        outputs = [
            cv2.resize(img, target_size[::-1], interpolation=resize_opt)
            for img in imgs
        ]

        if is_single_image:
            outputs = outputs[0]

        results['lq'] = outputs

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


allowed_degradations = {
    'RandomBlur': RandomBlur,
    # 'RandomResize': RandomResize,
    # 'RandomNoise': RandomNoise,
    'RandomJPEGCompression': RandomJPEGCompression,
    'RandomVideoCompression': RandomVideoCompression,
    'RandomScaleResize1': RandomScaleResize1,
}


class DegradationsWithShuffle1:
    """Apply random degradations to input, with degradations being shuffled.
    Degradation groups are supported. The order of degradations within the same
    group is preserved. For example, if we have degradations = [a, b, [c, d]]
    and shuffle_idx = None, then the possible orders are
    ::
        [a, b, [c, d]]
        [a, [c, d], b]
        [b, a, [c, d]]
        [b, [c, d], a]
        [[c, d], a, b]
        [[c, d], b, a]
    Modified keys are the attributed specified in "keys".
    Args:
        degradations (list[dict]): The list of degradations.
        keys (list[str]): A list specifying the keys whose values are
            modified.
        shuffle_idx (list | None, optional): The degradations corresponding to
            these indices are shuffled. If None, all degradations are shuffled.
    """

    def __init__(self, degradations, keys, shuffle_idx=None):

        self.keys = keys

        self.degradations = self._build_degradations(degradations)

        if shuffle_idx is None:
            self.shuffle_idx = list(range(0, len(degradations)))
        else:
            self.shuffle_idx = shuffle_idx

    def _build_degradations(self, degradations):
        for i, degradation in enumerate(degradations):
            if isinstance(degradation, (list, tuple)):
                degradations[i] = self._build_degradations(degradation)
            else:
                degradation_ = allowed_degradations[degradation['type']]
                degradations[i] = degradation_(degradation['params'],
                                               self.keys)

        return degradations

    def __call__(self, results):
        # shuffle degradations
        if len(self.shuffle_idx) > 0:
            shuffle_list = [self.degradations[i] for i in self.shuffle_idx]
            np.random.shuffle(shuffle_list)
            for i, idx in enumerate(self.shuffle_idx):
                self.degradations[idx] = shuffle_list[i]

        # apply degradations to input
        for degradation in self.degradations:
            if isinstance(degradation, (tuple, list)):
                for subdegrdation in degradation:
                    results = subdegrdation(results)
            else:
                results = degradation(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(degradations={self.degradations}, '
                     f'keys={self.keys}, '
                     f'shuffle_idx={self.shuffle_idx})')
        return repr_str
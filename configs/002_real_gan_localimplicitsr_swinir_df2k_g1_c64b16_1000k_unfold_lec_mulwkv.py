exp_name = '002_real_gan_localimplicitsr_swinir_df2k_g1_c64b16_1000k_unfold_lec_mulwkv'
scale_min, scale_max = 1, 4
gt_crop_size = 400
val_scale = 16

from mmedited.models.restorers.real_ciaosr import RealCiaoSR
from mmedited.models.backbones.sr_backbones.swinir_net import SwinIR
from mmedited.models.backbones.sr_backbones.ciaosr_net import LocalImplicitSRSWINIR
from mmedited.datasets.pipelines.random_degradations import RandomScaleResize1, DegradationsWithShuffle1, RandomBlur, RandomJPEGCompression
from mmedited.datasets.pipelines.crop import PairedRandomCropwScale
from mmedited.datasets.pipelines.generate_assistant import GenerateCoordinateAndCell1

# model settings
model = dict(
    type=RealCiaoSR,
    generator=dict(
        type=LocalImplicitSRSWINIR,
        window_size=8,
        encoder=dict(
            type=SwinIR,
            upscale=4,
            in_chans=3,
            img_size=48,
            window_size=8,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv',
            ),
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
            hidden_list=[256, 256, 256, 256]),
        feat_unfold=True,
        eval_bsize=30000,
        local_ensemble_coord=True,   #lec
        imnet_k_type='mul_w',
        imnet_v_type='mul_w',
        res=False,
        non_local_attn=False,
        cat_nla_v=False,
        ),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    rgb_mean=(0.4488, 0.4371, 0.4040),
    rgb_std=(1., 1., 1.),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=1e-1,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_sharpened_gt_in_pixel=True,
    is_use_sharpened_gt_in_percep=True,
    is_use_sharpened_gt_in_gan=False,
    is_use_ema=True)
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=[], crop_border=val_scale, scale=val_scale, tile=256, tile_overlap=32) #['PSNR', 'SSIM'] , convert_to='y'

# dataset settings
train_dataset_type = 'SRFolderGTDataset'
val_dataset_type = 'SRFolderGTDataset'
test_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(
        type='Crop',
        keys=['gt'],
        crop_size=(gt_crop_size, gt_crop_size),
        random_crop=True),
    dict(type='RescaleToZeroOne', keys=['gt']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    dict(type='CopyValues', src_keys=['gt_unsharp'], dst_keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 3],
            sigma_y=[0.2, 3],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2]),
        keys=['lq'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.2, 0.7, 0.1],  # up, down, keep
            resize_scale=[0.15, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0]),
        keys=['lq'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 30],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 3],
            poisson_gray_noise_prob=0.4),
        keys=['lq'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95]),
        keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            prob=0.8,
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 1.5],
            sigma_y=[0.2, 1.5],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2]),
        keys=['lq'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.3, 0.4, 0.3],  # up, down, keep
            resize_scale=[0.3, 1.2],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0]),
        keys=['lq'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 25],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 2.5],
            poisson_gray_noise_prob=0.4),
        keys=['lq'],
    ),
    dict(
        type=DegradationsWithShuffle1, 
        degradations=[
            dict(
                type='RandomJPEGCompression',
                params=dict(quality=[5, 50]),
            ),
            [
                dict(
                    type='RandomScaleResize1',
                    params=dict(
                        scale_min=scale_min,
                        scale_max=scale_max,
                        resize_opt=['bilinear', 'area', 'bicubic'],
                        resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
                ),
                dict(
                    type='RandomBlur',
                    params=dict(
                        prob=0.8,
                        kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                        kernel_list=['sinc'],
                        kernel_prob=[1],
                        omega=[3.1416 / 3, 3.1416]),
                ),
            ]
        ],
        keys=['lq'],
    ),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type=PairedRandomCropwScale, lq_patch_size=64),
    dict(type='Quantize', keys=['lq']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'gt_unsharp']),
    dict(type=GenerateCoordinateAndCell1, sample_quantity=4096, is_shuffle=False),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'gt_unsharp', 'coord', 'cell'],  
        meta_keys=['gt_path'])
]

valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='RandomDownSampling', scale_min=scale_max, scale_max=scale_max),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell'),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell', scale=scale_max),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]

real_pipeline = [
    # dict(
    #     type='LoadImageFromFile',
    #     io_backend='disk',
    #     key='gt',
    #     flag='color',
    #     channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='GenerateCoordinateAndCell', scale=val_scale),
    dict(
        type='Collect',
        keys=['lq', 'coord', 'cell'],
        meta_keys=['lq_path'])
]

data_dir = "data"
mydata_dir = "mydata"
data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=6, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=20,
        dataset=dict(
            type=train_dataset_type,
            gt_folder=f'{mydata_dir}/DF2K/HR',
            pipeline=train_pipeline,
            scale=scale_max)),
    val=dict(
        type=val_dataset_type,
        gt_folder=f'{mydata_dir}/Classical/Urban100/GTmod12',  #f'{data_dir}/testset/Urban100/HR',  #f'{data_dir}/sr_test/Urban100', #
        pipeline=valid_pipeline,
        scale=scale_max),
    test=dict(
        type=test_dataset_type,
        lq_folder=f'{mydata_dir}/RealSR/DPED',  #f'{mydata_dir}/RealSR/RealSRSet',  
        gt_folder=f'{mydata_dir}/RealSR/DPED',  #f'{mydata_dir}/RealSR/RealSRSet',
        pipeline=real_pipeline,
        scale=val_scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
iter_per_epoch = 1000
total_iters = 1000 * iter_per_epoch
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 600000, 800000],
    gamma=1)  #0.5

checkpoint_config = dict(interval=3000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=3000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None

# custom hook
custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.999),
    )
]

run_dir = './work_dirs'
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'{run_dir}/{exp_name}'
load_from = f'{run_dir}/002_real_wogan_localimplicitsr_swinir_df2k_g1_c64b16_1000k_unfold_lec_mulwkv/latest.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
test_checkpoint_path = f'{run_dir}/{exp_name}/latest.pth' # use --checkpoint None to enable this path in testing

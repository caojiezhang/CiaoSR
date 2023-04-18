exp_name = '001_localimplicitsr_swinir_div2k_g1_c64b16_1000k_unfold_lec_mulwkv_res_nonlocal'
scale_min, scale_max = 1, 4
val_scale = 4

from mmedited.models.restorers.ciaosr import CiaoSR
from mmedited.models.backbones.sr_backbones.swinir_net import SwinIR
from mmedited.models.backbones.sr_backbones.ciaosr_net import LocalImplicitSRSWINIR


# model settings
model = dict(
    type=CiaoSR,
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
        res=True,
        non_local_attn=True,
        cat_nla_v=True,
        ),
    rgb_mean=(0.4488, 0.4371, 0.4040),
    rgb_std=(1., 1., 1.),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
# test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=val_scale, scale=val_scale, convert_to='y')  #x6,x8,x12, B100
# test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=val_scale, tile=128, tile_overlap=32, scale=val_scale, convert_to='y')  #
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=val_scale, tile=128, tile_overlap=32, scale=val_scale)  # DIV2K
# test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=val_scale, scale=val_scale)  # DIV2K

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
        type='RandomDownSampling',
        scale_min=scale_min,
        scale_max=scale_max,
        patch_size=48),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell', sample_quantity=2304),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]

valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='RandomDownSampling', scale_min=val_scale, scale_max=val_scale),
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
    dict(type='GenerateCoordinateAndCell', scale=val_scale),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]

data_dir = "data"
mydata_dir = "mydata"
data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=10, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=20,
        dataset=dict(
            type=train_dataset_type,
            gt_folder=f'{data_dir}/DIV2K/HR',  #f'{data_dir}/DIV2K/DIV2K_train_HR', #
            pipeline=train_pipeline,
            scale=scale_max)),
    val=dict(
        type=val_dataset_type,
        gt_folder=f'{mydata_dir}/Classical/Urban100/GTmod12',  #f'{data_dir}/testset/Urban100/HR',  #f'{data_dir}/sr_test/Urban100', #
        pipeline=valid_pipeline,
        scale=scale_max),
    test=dict(
        # type=test_dataset_type,
        # lq_folder=f'{mydata_dir}/Classical/Manga109/LRbicx4',  #f'{mydata_dir}/Classical/Urban100/LRbicx3',  #f'{mydata_dir}/Classical/BSDS100/LRbicx4',  #f'{mydata_dir}/Classical/Set14/LRbicx4', #f'{data_dir}/testset/DIV2K_val/LR_bicubic/X2', #
        # gt_folder=f'{mydata_dir}/Classical/Manga109/GTmod12',  #f'{mydata_dir}/Classical/Urban100/GTmod12',  #f'{mydata_dir}/Classical/BSDS100/GTmod12',  #f'{mydata_dir}/Classical/Set14/GTmod12', #f'{data_dir}/testset/DIV2K_val/HR', #
        # pipeline=test_pipeline,
        # scale=val_scale,
        # filename_tmpl='{}'))   #x4
        type=val_dataset_type,
        gt_folder=f'{data_dir}/testset/DIV2K_val/HR', #f'{data_dir}/testset/Set5/HR', #f'{data_dir}/testset/B100/HR', #f'{data_dir}/testset/Manga109/HR',  #f'{data_dir}/testset/Urban100/HR',  #f'{data_dir}/testset/Set5/HR', #f'{mydata_dir}/Classical/BSDS100/GTmod12',  #f'{mydata_dir}/Classical/Set5/GTmod12', #f'{mydata_dir}/Classical/Manga109/GTmod12',  #f'{mydata_dir}/Classical/Urban100/GTmod12',  #f'{data_dir}/testset/Urban100/HR',  #
        pipeline=valid_pipeline,
        scale=val_scale))

# optimizer
optimizers = dict(type='Adam', lr=1.e-4)

# learning policy
iter_per_epoch = 1000
total_iters = 1000 * iter_per_epoch
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 600000, 800000],
    gamma=0.5)

checkpoint_config = dict(interval=3000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=3000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None

run_dir = './work_dirs'
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'{run_dir}/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
test_checkpoint_path = f'{run_dir}/{exp_name}/latest.pth' # use --checkpoint None to enable this path in testing

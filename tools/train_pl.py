# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
from mmcv import Config
from mmedit.datasets import build_dataset
from mmedit.models import build, MODELS

import socket
in_fbcode = True if 'fb' in socket.gethostname() or 'facebook' in socket.gethostname() else False

import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from mmedited.datasets.datamodule import DataModule

if in_fbcode:
    from iopath.common.file_io import PathManager
    from iopath.fb.manifold import ManifoldPathHandler
    pathmgr = PathManager()
    pathmgr.register_handler(ManifoldPathHandler())


def parse_args():
    parser = argparse.ArgumentParser(description="Train an editor")
    parser.add_argument("config", help="train config file path")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.gpus = torch.cuda.device_count()

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # automatic resume
    if cfg.resume_from is None:
        resume_from = cfg.work_dir + "/last.ckpt"
        if os.path.exists(resume_from):
            cfg.resume_from = resume_from
            print(f"automatic resuming from {resume_from}")

    def build_model(cfg, train_cfg=None, test_cfg=None, optimizers_cfg=None, lr_scheduler_cfg=None):
        """Build model.

        Args:
            cfg (dict): Configuration for building model.
            train_cfg (dict): Training configuration. Default: None.
            test_cfg (dict): Testing configuration. Default: None.
        """
        return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg,
                                       optimizers_cfg=optimizers_cfg,
                                       lr_scheduler_cfg=lr_scheduler_cfg))

    model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg,
                        optimizers_cfg=cfg.optimizers, lr_scheduler_cfg=cfg.lr_config)
    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]
    train_dataloader_cfg = {'batch_size': cfg.data.train_dataloader.samples_per_gpu*cfg.gpus, 'num_workers': cfg.data.workers_per_gpu, 'drop_last': cfg.data.train_dataloader.drop_last, 'pin_memory': True}
    val_dataloader_cfg = {'batch_size': cfg.data.val_dataloader.samples_per_gpu, 'num_workers': cfg.data.workers_per_gpu, 'pin_memory': True}
    datamodule = DataModule(train_dataset=datasets[0], val_dataset=datasets[1],
                            train_dataloader_cfg=train_dataloader_cfg, val_dataloader_cfg=val_dataloader_cfg)

    trainer = Trainer(default_root_dir=cfg.work_dir,
                      devices=cfg.gpus,
                      accelerator='auto',
                      num_sanity_val_steps=0,
                      strategy=DDPStrategy(find_unused_parameters=True, static_graph=False),
                      val_check_interval=cfg.evaluation['interval'],
                      check_val_every_n_epoch=None,
                      log_every_n_steps=cfg.log_config['interval'],
                      precision='bf16' if cfg.train_cfg['mixed_precision'] else 32,
                      max_steps=cfg.total_iters,
                      callbacks=[
                          ModelCheckpoint(every_n_train_steps=cfg.checkpoint_config.interval,
                                          monitor='val_eval_result_PSNR', mode='min',
                                          save_top_k=10, save_last=True, dirpath=cfg.work_dir),
                          EarlyStopping(monitor="val_eval_result_PSNR", patience=2, mode="min")])
    trainer.fit(model, datamodule, ckpt_path=cfg.resume_from)



if __name__ == "__main__":
    main()

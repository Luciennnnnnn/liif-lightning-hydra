import os
from typing import Any

import pytorch_lightning as pl
import torch

from torchvision.transforms import transforms

from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from skimage.metrics import peak_signal_noise_ratio

from ..super_resolution import super_resolution

def get_tensorboard_logger(trainer: pl.Trainer) -> TensorBoardLogger:
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, TensorBoardLogger):
            logger = lg

    if not logger:
        raise Exception(
            "You are using tensorboard related callback,"
            "but TensorBoardLogger was not found for some reason..."
        )

    return logger

class LogPSNRToTensorBoard(Callback):
    """log psnr."""

    def __init__(self):
        pass

    def on_test_batch_end(self, trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int):
        lr = batch['inp'][0]
        gt = batch['gt']
        gt_size = batch['gt_size'][0]

        preds = outputs['preds']

        pred = preds[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(gt_size[0].item(), gt_size[1].item(), 3).permute(2, 0, 1).cpu()
        
        gt = gt[0]
        gt = (gt * 0.5 + 0.5).clamp(0, 1).view(gt_size[0].item(), gt_size[1].item(), 3).permute(2, 0, 1).cpu()

        shave = 6 + 2 + dataloader_idx
        psnr = peak_signal_noise_ratio(gt[..., shave:-shave, shave:-shave].numpy(), pred[..., shave:-shave, shave:-shave].numpy(), data_range=1)

        if not os.path.isdir(os.path.join("results", "X" + str(dataloader_idx + 2))):
            os.makedirs(os.path.join("results", "X" + str(dataloader_idx + 2)))

        transforms.ToPILImage()(pred).save(os.path.join("results", "X" + str(dataloader_idx + 2), str(batch_idx) + ".png"))

        # log test metrics to your loggers!
        pl_module.log("test/psnr", psnr, on_step=False, on_epoch=True)

    def on_validation_batch_end(self, trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int):
        if batch_idx % 3 == 0:
            logger = get_tensorboard_logger(trainer=trainer)
            tensorboard = logger.experiment

            lr = batch['inp'][0]
            gt_size = batch['gt_size'][0]
            sr = outputs['sr']

            sr = (sr * 0.5 + 0.5).clamp(0, 1).view(gt_size[0].item(), gt_size[1].item(), 3).permute(2, 0, 1)
            lr = (lr * 0.5 + 0.5).clamp(0, 1)

            tensorboard.add_image("val/lr", lr, global_step=trainer.global_step)
            tensorboard.add_image("val/sr", sr, global_step=trainer.global_step)
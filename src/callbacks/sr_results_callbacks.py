import os
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

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
        logger = get_tensorboard_logger(trainer=trainer)
        experiment = logger.experiment

        lr = batch['inp'][0]
        gt = batch['gt']
        gt_size = batch['gt_size'][0]

        preds = super_resolution(model=pl_module, x=lr.unsqueeze[0], target_resolution=gt_size, bsize=30000)

        pred = preds[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(gt_size[0].item(), gt_size[1].item(), 3).permute(2, 0, 1).cpu()

        gt = gt[0]
        gt = (gt * 0.5 + 0.5).clamp(0, 1).view(gt_size[0].item(), gt_size[1].item(), 3).permute(2, 0, 1).cpu()

        shave = 6 + 2 + dataset_idx
        psnr = peak_signal_noise_ratio(gt[..., shave:-shave, shave:-shave].numpy(), pred[..., shave:-shave, shave:-shave].numpy(), data_range=1)

        print("asdasdasdasdasdasda")
        print("qweqweqweqweqweqwe")
        
        if not os.path.isdir(os.path.join("results", "X" + str(dataset_idx + 2))):
            os.makedirs(os.path.join("results", "X" + str(dataset_idx + 2)))

        transforms.ToPILImage()(pred).save(os.path.join("results", "X" + str(dataset_idx + 2), str(batch_idx) + ".png"))

        # log test metrics to your loggers!
        pl_module.log("test/psnr", psnr, on_step=False, on_epoch=True)
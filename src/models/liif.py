import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

from pytorch_lightning.metrics.classification import Accuracy
from torch.optim import Optimizer

from skimage.metrics import peak_signal_noise_ratio

from ..architectures.simple_dense_net import SimpleDenseNet
from ..utils.functional import make_coord
from ..super_resolution import super_resolution

class LIIF(pl.LightningModule):
    """
    LightningModule of LIIF for SR.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Links:
        https://github.com/yinboc/liif
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # loss function
        self.criterion = torch.nn.L1Loss()
        
        self.encoder = hydra.utils.instantiate(self.hparams.encoder, _recursive_=False)

        inr_in_dim = self.encoder.out_dim
        if self.hparams.feat_unfold:
            inr_in_dim *= 9
        inr_in_dim += 2 # attach coord
        if self.hparams.cell_decode:
            inr_in_dim += 2

        self.INR = hydra.utils.instantiate(self.hparams.inr, in_dim=inr_in_dim)

        self.metric_hist = {
            "train/loss": [],
            "val/loss": [],
        }

    def look_up_feature(self, coordinate, feature, feat_coord):
        """
        Args:
            coordinate (Tensor): N×Q×2
            feature (Tensor): N×C×H×W
        Returns:
            tuple: (feature: N×Q×C, position: N×Q×2) corresponding feature and coordinate.
        """

        # 对应位置的feature, 取空间上邻近的
        feature = F.grid_sample(
                    feature, coordinate.flip(-1).unsqueeze(1), # N×1×Q×2
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # N×Q×C

        # 每个feature实际的坐标
        f_coordinate = F.grid_sample(
            feat_coord, coordinate.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # N×Q×2

        return feature, f_coordinate

    def forward(self, feature, coord, gt_size) -> torch.Tensor:
    
        if self.hparams.feat_unfold:
            feature = F.unfold(feature, 3, padding=1).view(
                feature.shape[0], feature.shape[1] * 9, feature.shape[2], feature.shape[3])

        # field radius (global: [-1, 1])
        rx = 2 / feature.shape[-2] / 2
        ry = 2 / feature.shape[-1] / 2

        feat_coord = make_coord(feature.shape[-2:], flatten=False).type_as(feature) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feature.shape[0], 2, *feature.shape[-2:]) # N×2×H×W
            
        if self.hparams.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone() #  N×Q×2
                # left-top, left-down, right-top, right-down move one radius.
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat, q_coord = self.look_up_feature(coord_, feature, feat_coord)
                
                # 相对于feature位置的偏移量
                relative_offset = coord - q_coord
                relative_offset[:, :, 0] *= feature.shape[-2]
                relative_offset[:, :, 1] *= feature.shape[-1]

                area = torch.abs(relative_offset[:, :, 0] * relative_offset[:, :, 1])

                inp = torch.cat([q_feat, relative_offset], dim=-1)

                if self.hparams.cell_decode:
                    cell = (2 / (gt_size.unsqueeze(1).repeat(1, coord.shape[1], 1))).type_as(inp)

                    cell[:, :, 0] *= feature.shape[-2]
                    cell[:, :, 1] *= feature.shape[-1]
                    inp = torch.cat([inp, cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.INR(inp.view(bs * q, -1)).view(bs, q, -1)

                preds.append(pred)
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.hparams.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def step(self, batch) -> Dict[str, torch.Tensor]:
        x, coord, gt, gt_size = batch['inp'], batch['coord'], batch['gt'], batch['gt_size']
        feature = self.encoder(x) # N×C×H×W
        pred = self.forward(feature, coord, gt_size)
        loss = self.criterion(pred, gt)
        return loss, pred, gt

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)

        # log train metrics to your loggers!
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        # return {"loss": loss, "preds": preds, "targets": targets}
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)
        # log val metrics to your loggers!
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        lr = batch['inp'][0]
        gt_size = batch['gt_size'][0]
        sr = super_resolution(model=self, x=lr.unsqueeze(0), target_resolution=gt_size.cpu(), bsize=30000)[0]

        # we can return here dict with any tensors
        # and then read it in some callback or in validation_epoch_end() below
        # return {"loss": loss, "preds": preds, "targets": targets, "sr": sr}
        return {"loss": loss, "sr": sr}

    def test_step(self, batch: Any, batch_idx: int, dataset_idx: int) -> Dict[str, torch.Tensor]:
        lr = batch['inp'][0]
        gt = batch['gt']
        gt_size = batch['gt_size'][0]
        preds = super_resolution(model=self, x=lr.unsqueeze(0), target_resolution=gt_size.cpu(), bsize=30000)

        loss = self.criterion(gt, preds)

        pred = preds[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(gt_size[0].item(), gt_size[1].item(), 3).permute(2, 0, 1).cpu()
        
        gt = gt[0]
        gt = (gt * 0.5 + 0.5).clamp(0, 1).view(gt_size[0].item(), gt_size[1].item(), 3).permute(2, 0, 1).cpu()

        shave = 6 + 2 + dataset_idx
        psnr = peak_signal_noise_ratio(gt[..., shave:-shave, shave:-shave].numpy(), pred[..., shave:-shave, shave:-shave].numpy(), data_range=1)

        if not os.path.isdir(os.path.join("results", "X" + str(dataset_idx + 2))):
            os.makedirs(os.path.join("results", "X" + str(dataset_idx + 2)))

        transforms.ToPILImage()(pred).save(os.path.join("results", "X" + str(dataset_idx + 2), str(batch_idx) + ".png"))

        # log test metrics to your loggers!
        self.log("test/psnr", psnr, on_step=False, on_epoch=True)

        # log test metrics to your loggers!
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}

    # [OPTIONAL METHOD]
    def training_epoch_end(self, outputs: List[Any]) -> None:
        # log best so far train acc and train loss
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    # [OPTIONAL METHOD]
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # log best so far val acc and val loss
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        """
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        lr_scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler, optim
        )
        return [optim], [lr_scheduler]
import torch

from pytorch_lightning import LightningModule
from .utils.functional import make_coord

def super_resolution(model: LightningModule, x, target_resolution, bsize):
        coord = make_coord(target_resolution.cpu()).type_as(x).unsqueeze(0)
        target_resolution = target_resolution.unsqueeze(0)

        with torch.no_grad():
            feature = model.encoder(x)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = model(feature, coord[:, ql: qr, :], target_resolution)
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred
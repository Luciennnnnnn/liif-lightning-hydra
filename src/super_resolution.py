from PIL import Image

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from pytorch_lightning import LightningModule
from .utils import functional

def super_resolution(model: LightningModule, x, target_resolution, bsize):
        coord = functional.make_coord(target_resolution).type_as(x).unsqueeze(0)
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

# @hydra.main()
# def main(config: DictConfig):
#     from .models.liif import LIIF
#     from torchvision.transforms import transforms

#     # load Lightning model
#     print(f"Loading model from <{config.CKPT_PATH}>")
#     model = LIIF.load_from_checkpoint(checkpoint_path=config.CKPT_PATH)

#     # print model hyperparameters
#     print(model.hparams)

#     img = transforms.ToTensor()(Image.open(os.path.join(config.input).convert('RGB'))
#     h, w = config.resolution
#     pred = super_resolution(model=model, x=img.unsqueeze(0), target_resolution=(h, w), bsize=30000)[0]
        
#     pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
#     transforms.ToPILImage()(pred).save(config.output)

# if __name__ == "__main__":
#     main()
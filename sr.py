import os
from PIL import Image

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from pytorch_lightning import LightningModule


@hydra.main()
def main(config: DictConfig):
    from src.models.liif import LIIF
    from torchvision.transforms import transforms
    from src.super_resolution import super_resolution
    # load Lightning model
    print(f"Loading model from <{config.CKPT_PATH}>")
    model = LIIF.load_from_checkpoint(checkpoint_path=config.CKPT_PATH)
    model.eval()
    model.freeze()
    # print model hyperparameters
    model = model.cuda()

    image_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    img = image_transforms(Image.open(hydra.utils.to_absolute_path(config.input)).convert('RGB'))
    h, w = config.resolution
    print(f"{h}, {w}")
    pred = super_resolution(model=model, x=img.unsqueeze(0).cuda(), target_resolution=torch.Tensor([h, w]), bsize=30000)[0]
        
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(hydra.utils.to_absolute_path(config.output))

if __name__ == "__main__":
    main()

python sr.py +input=0831x2.png +resolution=[1356,2040] +output=0831_sr2.png +CKPT_PATH='/gdata/luoxin/lightning-hydra-template/logs/runs/2021-03-24/02-58-29/checkpoints/last.ckpt'

python sr.py +input=0831x3.png +resolution=[1356,2040] +output=0831_sr3.png +CKPT_PATH='/gdata/luoxin/lightning-hydra-template/logs/runs/2021-03-24/02-58-29/checkpoints/last.ckpt'

python sr.py +input=0831x4.png +resolution=[1356,2040] +output=0831_sr4.png +CKPT_PATH='/gdata/luoxin/lightning-hydra-template/logs/runs/2021-03-24/02-58-29/checkpoints/last.ckpt'

python demo.py --input 0831x2.png --model edsr-baseline-liif.pth --resolution 1356,2040 --output 0831_sr2.png --gpu 0
python demo.py --input 0831x3.png --model edsr-baseline-liif.pth --resolution 1356,2040 --output 0831_sr3.png --gpu 0
python demo.py --input 0831x4.png --model edsr-baseline-liif.pth --resolution 1356,2040 --output 0831_sr4.png --gpu 0

python demo.py --input 0831x2.png --model rdn-liif.pth --resolution 1356,2040 --output 0831_sr2rdn.png --gpu 0
python demo.py --input 0831x3.png --model rdn-liif.pth --resolution 1356,2040 --output 0831_sr3rdn.png --gpu 0
python demo.py --input 0831x4.png --model rdn-liif.pth --resolution 1356,2040 --output 0831_sr4rdn.png --gpu 0

python demo.py --input 0831.png --model edsr-baseline-liif.pth --resolution 339,510 --output 0831_lr4.png --gpu 0,1
python demo.py --input 0831.png --model rdn-liif.pth --resolution 339,510 --output 0831_lr4rdn.png --gpu 0,1
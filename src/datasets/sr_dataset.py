import os
from enum import Enum
from typing import Callable, Optional, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset

class SRDataset(Dataset):
    """Example dataset class for loading images from folder."""

    def __init__(self, dataset: Dataset, lr_size: Tuple[int, int], transform: Optional[Callable] = None):
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset

    def __getitem__(self, idx):
        # image = Image.open(self.images[idx]).convert("L")  # convert to black and white
        # read from folder
        hr = self.dataset[idx]

        if self.transform: # crop image
            hr = self.transform(hr)

        lr = transforms.Resize(lr_size, interpolation=InterpolationMode.BICUBIC)(hr)

        return lr, hr

    def __len__(self):
        return len(self.dataset)


class ContinuesSRDataset(Dataset):
    """Example dataset class for loading images from folder."""

    def __init__(self, dataset: Dataset, inp_size, scale_range: Union(Union(int, float), Tuple[Union(int, float), Union(int, float)])):
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset

        if type(scale_range) == float or type(scale_range) == int:
            self.scale_range = (scale_range[0], scale_range[1])
        else if type(scale_range) == Tuple:
            assert(scale_range[0] <= scale_range[1])
            self.scale_range = scale_range

        if inp_size is not None:
            if not isinstance(inp_size, (int, Sequence)):
                raise TypeError("inp_size should be int or sequence. Got {}".format(type(inp_size)))
            if isinstance(inp_size, Sequence) and len(inp_size) not in (1, 2):
                raise ValueError("If inp_size is a sequence, it should have 1 or 2 values")

            if isinstance(inp_size, int):
                inp_size = (inp_size, inp_size)
        
        self.inp_size = inp_size
        
        if not isinstance(scale_range, (int, float, Sequence)):
            raise TypeError("inp_size should be int or sequence. Got {}".format(type(scale_range)))
        if isinstance(scale_range, Sequence) and len(scale_range) not in (1, 2):
            raise ValueError("If inp_size is a sequence, it should have 1 or 2 values")

        if isinstance(scale_range, (int, float)):
            scale_range = (scale_range, scale_range)

        self.scale_range = scale_range

    def __getitem__(self, idx):
        hr = self.dataset[idx]
        
        s = random.uniform(self.scale_range[0], self.scale_range[1])
        if self.inp_size:
            h_lr, w_lr = self.inp_size
        else:
            h_lr = math.floor(hr.shape[-2] / s + 1e-9)
            w_lr = math.floor(hr.shape[-1] / s + 1e-9)
            
        h_hr, w_hr = round(h_lr * s), round(w_lr * s)
        
        hr = transforms.RandomCrop((h_hr, w_hr))(hr)
        lr = transforms.Resize((h_lr, w_lr), interpolation=InterpolationMode.BICUBIC)(hr)

        return lr, hr

    def __len__(self):
        return len(self.dataset)
import os
from enum import Enum
from typing import Callable, Optional
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """Example dataset class for loading images from folder."""

    def __init__(self, dir: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = [os.path.join(dir, fname) for fname in os.listdir(dir)]

    def __getitem__(self, idx):
        # image = Image.open(self.images[idx]).convert("L")  # convert to black and white
        image = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)

class DIV2KDataset(Dataset):
    """Example dataset class for loading images from folder."""

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        self.transform = transform

        if train:
            target_dir = os.path.join(root, "DIV2K_train_HR")
        else:
            target_dir = os.path.join(root, "DIV2K_valid_HR")
        
        self.images = [os.path.join(target_dir, fname) for fname in os.listdir(target_dir)]

    def __getitem__(self, idx):
        # image = Image.open(self.images[idx]).convert("L")  # convert to black and white

        # read from folder
        image = Image.open(self.images[idx]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)

# class DIV2KDataset2(Dataset):
#     """Example dataset class for loading images from folder."""

#     def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, scale = 2):
#         self.transform = transform
#         self.target_transform = target_transform
        
#         if train:
#             input_dir = os.path.join(root, "DIV2K_train_LR_bicubic")
#             target_dir = os.path.join(root, "DIV2K_train_HR")
#         else:
#             input_dir = os.path.join(root, "DIV2K_valid_LR_bicubic")
#             target_dir = os.path.join(root, "DIV2K_valid_HR")
        
#         if not os.path.exists(os.path.join(input_dir, f"X{scale}")):
#             raise FileNotFoundError(f"Can not find X{scale} downsampled images in {os.path.join(input_dir, f"X{scale}")}")

#         self.input_images = [os.path.join(input_dir, f"X{scale}", fname) for fname in os.listdir(os.path.join(input_dir, f"X{scale}"))]
#         self.target_images = [os.path.join(target_dir, fname) for fname in os.listdir(target_dir)]

#     def __getitem__(self, idx):
#         # image = Image.open(self.images[idx]).convert("L")  # convert to black and white
#         # read from folder

#         _input = Image.open(self.input_images[idx]).convert("RGB")
#         target = Image.open(self.target_images[idx]).convert("RGB")
        
#         if self.transform:
#             _input = self.transform(_input)

#         if self.target_transform:
#             target = self.target_transform(target)

#         return _input, target

#     def __len__(self):
#         return len(self.input_images)


# for size-varied ground-truths experienment.
class CelebAHQDataset(Dataset):
    """Example dataset class for loading images from folder."""

    def __init__(self, dir: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = [os.path.join(dir, fname) for fname in os.listdir(dir)]

    def __getitem__(self, idx):
        # image = Image.open(self.images[idx]).convert("L")  # convert to black and white
        image = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)
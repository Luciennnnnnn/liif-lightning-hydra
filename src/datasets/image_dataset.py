import os
from enum import Enum
from typing import Callable, Optional
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    """Dataset representing a folder of images.
    Args:
        sorted:
        sample_indices:
        first_k:
        filenames:
    """

    def __init__(self, root: str, is_sort=True, sample_indices=None, first_k=None,
                filenames=None,
                transform: Optional[Callable] = None, 
                target_transform: Optional[Callable] = None):
        self.transform = transform
        self.target_transform = target_transform

        if filenames is None:
            filenames = os.listdir(root)

            if sample_indices:
                filenames = filenames[sample_indices]

        if is_sort:
            filenames = sorted(filenames)

        if first_k:
            filenames = filenames[:first_k]

        self.images = [os.path.join(root, fname) for fname in filenames]

    def __getitem__(self, idx):
        # image = Image.open(self.images[idx]).convert("L")  # convert to black and white
        image = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)

class DIV2KDataset(Dataset):
    """Dataset of DIV2K for SR used on training."""

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        self.transform = transform

        if train:
            target_dir = os.path.join(root, "DIV2K_train_HR")
        else:
            target_dir = os.path.join(root, "DIV2K_valid_HR")
        
        self.dataset = ImageFolderDataset(root=target_dir, is_sort=True, transform=transform)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class DIV2KDatasetPaired(Dataset):
    """Dataset of DIV2K for SR used on validation.
    Args:
        scale:
        shared_transform:
    """

    VALID_LR_FOLDER = "DIV2K_valid_LR_bicubic"
    VALID_HR_FOLDER = "DIV2K_valid_HR"

    def __init__(self, root: str, scale = 2, shared_transform: Optional[Callable] = None):
        self.shared_transform = shared_transform

        input_dir = os.path.join(root, self.VALID_LR_FOLDER, f"X{scale}")
        target_dir = os.path.join(root, self.VALID_HR_FOLDER)
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Can not find X{scale} downsampled images in {input_dir}")

        self.dataset_1 = ImageFolderDataset(root=input_dir, is_sort=True)
        self.dataset_2 = ImageFolderDataset(root=target_dir, is_sort=True)

    def __getitem__(self, idx):
        _input = self.dataset_1[idx]
        target = self.dataset_2[idx]
        
        if self.shared_transform:
            _input = self.shared_transform(_input)
            target = self.shared_transform(target)
        return _input, target

    def __len__(self):
        return len(self.dataset_1)

# for size-varied ground-truths experienment.
class CelebAHQDataset(Dataset):
    """Dataset of CelebA-HQ for SR.
    Args:
        lr_size:
        hr_size:
        split_file:
        first_k:
        shared_transform:
    """

    def __init__(self, root: str, lr_size, hr_size, split_file=None, first_k=None, shared_transform: Optional[Callable] = None):
        self.shared_transform = shared_transform

        input_dir = os.path.join(root, str(lr_size))
        target_dir = os.path.join(root, str(hr_size))

        if split_file:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]            

        self.dataset_1 = ImageFolderDataset(root=input_dir, is_sort=False, filenames=filenames, first_k=first_k)
        self.dataset_2 = ImageFolderDataset(root=target_dir, is_sort=False, filenames=filenames, first_k=first_k)

    def __getitem__(self, idx):
        _input = self.dataset_1[idx]
        target = self.dataset_2[idx]
        
        if self.shared_transform:
            _input = self.shared_transform(_input)
            target = self.shared_transform(target)
        return _input, target

    def __len__(self):
        return len(self.dataset_1)
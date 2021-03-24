from typing import Optional, Sequence

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

from ..datasets.image_dataset import CelebAHQDataset
from ..datasets.sr_dataset import SRImplicitUniformVaried
from ..datasets.implicit_image_dataset import ImplicitImageDataset

from ..utils.transforms import RandomDFlip, RandomHorizontalFlipList

class CelebAHQDataModule(LightningDataModule):
    """
    LightningDataModule for CelebAHQ dataset

    https://github.com/tkarras/progressive_growing_of_gans
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.data_dir = kwargs["data_dir"]

        self.split_file = kwargs["split_file"]
        self.first_k = kwargs["first_k"]

        self.train_dataloader_params = kwargs["train_dataloader_params"]
        self.val_dataloader_params = kwargs["val_dataloader_params"]
        self.test_dataloader_params = kwargs["test_dataloader_params"]

        self.lr_size = kwargs["lr_size"]
        self.hr_size = kwargs["hr_size"]
        self.sample_q = kwargs["sample_q"]

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # Can write a download program to replace manual dowload 
        pass


    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val."""

        # train dataset
        trainset = CelebAHQDataset(root=self.data_dir, train=True, lr_size=self.lr_size, hr_size=self.hr_size, split_file=self.split_file, transform=self.transforms, shared_transform=RandomHorizontalFlipList(p=0.5))
        trainset = SRImplicitUniformVaried(dataset=trainset, size_min=self.lr_size, size_max=self.hr_size)
        trainset = ImplicitImageDataset(dataset=trainset, sample_q=self.sample_q)

        # validation dataset
        valset = CelebAHQDataset(root=self.data_dir, train=False, lr_size=self.lr_size, hr_size=self.hr_size, split_file=self.split_file, first_k=self.first_k, transform=self.transforms)
        valset = SRImplicitUniformVaried(dataset=valset, size_min=self.lr_size, size_max=self.hr_size)
        valset = ImplicitImageDataset(dataset=valset, sample_q=self.sample_q)

        self.data_train, self.data_val = trainset, valset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_dataloader_params.batch_size,
            num_workers=self.train_dataloader_params.num_workers,
            pin_memory=self.train_dataloader_params.pin_memory,
            shuffle=self.train_dataloader_params.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_dataloader_params.batch_size,
            num_workers=self.val_dataloader_params.num_workers,
            pin_memory=self.val_dataloader_params.pin_memory,
            shuffle=self.val_dataloader_params.shuffle,
        )

    def test_dataloader(self):
        testset = CelebAHQDataset(root=self.data_dir, train=False, lr_size=self.lr_size, hr_size=self.hr_size, split_file=self.split_file, transform=self.transforms)
        testset = ImplicitImageDataset(dataset=valset, sample_q=self.sample_q)

        test_dataloader = DataLoader(
            dataset=testset,
            batch_size=self.test_dataloader_params.batch_size,
            num_workers=self.test_dataloader_params.num_workers,
            pin_memory=self.test_dataloader_params.pin_memory,
            shuffle=self.test_dataloader_params.shuffle,
        )
        return test_dataloader
from typing import Optional, Sequence

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

from ..datasets.image_dataset import DIV2KDataset
from ..datasets.sr_dataset import ContinuesSRDataset
from ..datasets.implicit_image_dataset import ImplicitImageDataset

from ..utils.transforms import RandomDFlip

class DIV2KDataModule(LightningDataModule):
    """
    LightningDataModule for DIV2K dataset, only

    https://data.vision.ee.ethz.ch/cvl/DIV2K/
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.data_dir = kwargs["data_dir"]
        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]

        self.inp_size = kwargs["inp_size"]
        self.scale_range = kwargs["scale_range"]
        self.sample_q = kwargs["sample_q"]

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        # self.dims is returned when you call datamodule.size()
        # self.dims = (1, 28, 28)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        # Can write a download program to replace dowload 
        pass


    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val."""

        # 1. augment
        augument_transforms = transforms.Compose([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomDFlip(p=0.5)])

        # train dataset
        trainset = DIV2KDataset(root=self.data_dir, train= True, transform=transforms.Compose([self.transforms, augument_transforms]))
        trainset = ContinuesSRDataset(dataset=trainset, inp_size=self.inp_size, scale_range=self.scale_range)
        trainset = ImplicitImageDataset(dataset=trainset, sample_q=self.sample_q)

        # validation dataset
        valset = DIV2KDataset(root=self.data_dir, train=False, transform=self.transforms)
        valset = ContinuesSRDataset(dataset=valset, inp_size=self.inp_size, scale_range=self.scale_range)
        valset = ImplicitImageDataset(dataset=valset, sample_q=self.sample_q)

        self.data_train, self.data_val = trainset, valset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         shuffle=False,
    #     )

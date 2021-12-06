import os.path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


NUM_WORKERS = os.cpu_count() // 2


class CIFARDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = './datasets',
            num_workers: int = NUM_WORKERS,
            *,
            batch_size: int
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar10_train = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True)

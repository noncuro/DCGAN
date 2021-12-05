from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data import CIFARDataModule
from gan import GAN

wandb_logger = WandbLogger(project="CIFAR_GAN")

LATENT_DIM = 128
BATCH_SIZE = 128
data_dir = './datasets'

if __name__ == "__main__":
    dm = CIFARDataModule(data_dir=data_dir, batch_size=BATCH_SIZE)
    model = GAN(latent_dim=LATENT_DIM, batch_size=BATCH_SIZE)
    trainer = Trainer(logger=wandb_logger, log_every_n_steps=2)
    trainer.fit(model, dm)

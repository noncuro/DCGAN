import torch.cuda
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data import CIFARDataModule
from gan import GAN

wandb_logger = WandbLogger(project="CIFAR_GAN")

LATENT_DIM = 128
BATCH_SIZE = 32

AVAIL_GPUS = torch.cuda.device_count()

data_dir = './datasets'

if __name__ == "__main__":
    dm = CIFARDataModule(data_dir=data_dir, batch_size=BATCH_SIZE)
    model = GAN(latent_dim=LATENT_DIM, batch_size=BATCH_SIZE)
    wandb_logger.watch(model)
    trainer = Trainer(logger=wandb_logger,
                      log_every_n_steps=2,
                      gpus=-1,
                      accelerator="ddp",
                      precision=16,
                      # gradient_clip_val=1
                      )
    trainer.fit(model, dm)

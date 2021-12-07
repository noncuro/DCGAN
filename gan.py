import os.path
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torchvision
import wandb
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from torchvision.utils import save_image

from img_helpers import *
from model import *


class GAN(LightningModule):
    def __init__(
            self,
            latent_dim: int = 32,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            use_weights_init: bool = True,
            *,
            batch_size: int,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = Generator(self.hparams.latent_dim)
        self.discriminator = Discriminator()

        if self.hparams.use_weights_init:
            self.discriminator.apply(weights_init)
            self.generator.apply(weights_init)

        # Fixed noise, also for pt lightning
        self.example_input_array = torch.randn(8, self.hparams.latent_dim)

        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # train generator
        if optimizer_idx == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
            z = z.type_as(imgs)

            # generate images
            generated_imgs = self(z)
            self.generated_imgs = generated_imgs.detach()

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            disc_pred_on_fake = self.discriminator(generated_imgs)
            g_loss = self.adversarial_loss(disc_pred_on_fake, valid)
            disc_pred_on_fake = disc_pred_on_fake.detach().mean().item()
            tqdm_dict = {"g_loss": g_loss.item(), "disc_pred_on_fake": disc_pred_on_fake}
            self.log("g_loss", g_loss.item())
            self.log("disc_pred_on_fake", disc_pred_on_fake)
            # self.logger.log_metrics(tqdm_dict, step=self.global_step)
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            if batch_idx % 2 == 1:
                return
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            real_label = torch.ones(imgs.size(0), 1)
            real_label = real_label.type_as(imgs)

            disc_pred_on_real = self.discriminator(imgs)
            real_loss = self.adversarial_loss(disc_pred_on_real, real_label)

            disc_pred_on_real = disc_pred_on_real.detach().mean().item()

            # how well can it label as fake?
            fake_label = torch.zeros(imgs.size(0), 1)
            fake_label = fake_label.type_as(imgs)
            # Reuse images last generated
            fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), fake_label)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) * 0
            tqdm_dict = {"d_loss": d_loss.item(), "disc_pred_on_real": disc_pred_on_real}
            self.log_dict(tqdm_dict)
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self) -> None:
        if self.trainer.is_global_zero:
            with torch.no_grad():
                z = self.example_input_array.type_as(self.generator.initial_layers[0].weight)
                generated_imgs = self(z)
            grid = torchvision.utils.make_grid(generated_imgs)
            wandb.log({"sample_imgs": wandb.Image(grid), "trainer/global_step": self.trainer.global_step})

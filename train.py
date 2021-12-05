import os.path
from typing import Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from torchvision.utils import save_image

from img_helpers import *
from model import *


REAL_LABEL = 1
FAKE_LABEL = 0

NUM_TRAIN = 49000

num_epochs = 100
learning_rate = 3e-4
latent_vector_size = 32
use_weights_init = True

# device selection
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

# We set a random seed to ensure that your results are reproducible.
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)




def get_loader_train(batch_size):
    return DataLoader(cifar10_train,
                      batch_size=batch_size,
                      collate_fn=create_collate_fn(batch_size, latent_vector_size=latent_vector_size, device=device),
                      sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                      drop_last=True)


def train(checkpoint=0):
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    model_G = Generator(latent_vector_size).to(device)
    if use_weights_init:
        model_G.apply(weights_init)

    model_D = Discriminator().to(device)
    if use_weights_init:
        model_D.apply(weights_init)

    loss_function = nn.BCELoss(reduction='mean')

    beta1 = 0.5
    optimizerD = torch.optim.Adam(model_D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device)

    wandb.init(mode="disabled")
    global_step = -1

    BATCH_SIZE = 256

    for epoch in range(checkpoint + 1, checkpoint + num_epochs + 1):
        model_D.train()
        model_G.train()
        loader_train = get_loader_train(BATCH_SIZE)
        for (real_x, real_label), (noise, fake_label) in loader_train:
            global_step += 1
            train_loss_D = 0
            train_loss_G = 0

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################device
            # train with real
            optimizerD.zero_grad()

            output = model_D(real_x)
            disc_loss_on_real = loss_function(output, real_label.float())
            disc_loss_on_real.backward()
            average_disc_pred_on_real = output.detach().mean().item()
            wandb.log(dict(average_disc_pred_on_real=average_disc_pred_on_real,
                           disc_loss_on_real=disc_loss_on_real), step=global_step)

            # train with fake
            fake = model_G(noise)
            disc_pred_on_fake = model_D(fake.detach())
            disc_loss_on_fake = loss_function(disc_pred_on_fake, fake_label.float())
            disc_loss_on_fake.backward()
            avg_disc_pred_on_fake = disc_pred_on_fake.mean().item()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            optimizerG.zero_grad()
            disc_pred_on_fake = model_D(fake) # Is it bad that we're computing this again?
            gen_loss = loss_function(disc_pred_on_fake, real_label.float())
            gen_loss.backward()

            optimizerG.step()
            wandb.log(dict(gen_loss=gen_loss), step=global_step)
            if epoch == 0:
                temp = denorm(real_x.cpu()).float()
                save_image(temp, './saved_models/real_samples.png')
                wandb.log({"real_samples": wandb.Image(temp)}, step=global_step)

            model_D.eval()
            model_G.eval()
            with torch.no_grad():
                fake = model_G(fixed_noise)
                temp = denorm(fake.cpu()).float()
                save_image(temp, './saved_models/fake_samples_epoch_%03d.png' % epoch)
                wandb.log({"fake_samples": wandb.Image(temp)}, step=global_step)
        # SAVE CHECKPOINT
        torch.save(model_G.state_dict(), f"saved_models/DCGAN_model_G_{epoch}.pth")
        torch.save(model_D.state_dict(), f"saved_models/DCGAN_model_D_{epoch}.pth")
        print(f"Saved after epoch: {epoch}")


if __name__ == "__main__":
    train()

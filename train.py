import torch
from torch.utils.data import DataLoader
from chest_xray_dataset import ChestXrayDataset
from torch import nn
from torch import mps
from Models.generator import Generator
from Models.discriminator import Discriminator
import gc
import torch.multiprocessing as tmp
from torchvision.utils import save_image
import wandb


def train_step():
    E_reals = 0
    E_fakes = 0

    total_gen_loss = 0
    total_disc_loss = 0

    for step, sample in enumerate(train_loader):
        sample = sample.to(device=device)

        fake_e = gen_E(sample)
        disc_e_real = disc_E(sample)
        disc_e_fake = disc_E(fake_e.detach())

        E_reals += disc_e_real.mean().item()
        E_fakes += disc_e_fake.mean().item()

        disc_e_real_loss = mse(disc_e_real, torch.ones_like(disc_e_real))
        disc_e_fake_loss = mse(disc_e_fake, torch.zeros_like(disc_e_fake))

        disc_e_loss = disc_e_real_loss+disc_e_fake_loss

        fake_d = gen_D(sample)
        disc_d_real = disc_D(sample)
        disc_d_fake = disc_D(fake_d.detach())

        disc_d_real_loss = mse(disc_d_real, torch.ones_like(disc_d_real))
        disc_d_fake_loss = mse(disc_d_fake, torch.zeros_like(disc_d_fake))

        disc_d_loss = disc_d_real_loss+disc_d_fake_loss

        disc_loss = (disc_e_loss+disc_d_loss)/2

        # Training the Discriminator
        disc_opt.zero_grad()
        disc_loss.backward()

        disc_opt.step()

        disc_e_fake = disc_E(fake_e)
        disc_d_fake = disc_D(fake_d)
        loss_G_e = mse(disc_e_fake, torch.ones_like(disc_e_fake))
        loss_G_d = mse(disc_d_fake, torch.ones_like(disc_d_fake))

        cycle_encrypted = gen_E(fake_d)
        cycle_decrypted = gen_D(fake_e)

        cycle_enc_loss = l1(sample, cycle_encrypted)
        cycle_dec_loss = l1(sample, cycle_decrypted)

        gen_loss = (loss_G_e+loss_G_d+cycle_enc_loss*10+cycle_dec_loss*10)

        gen_opt.zero_grad()
        gen_loss.backward()

        gen_opt.step()

        total_gen_loss += gen_loss.mean().item()
        total_disc_loss += disc_loss.mean().item()

        # Save output
        if step == steps-1:
            save_image(fake_d*0.5+0.5, f"outputs/Decrypted/dec_{step}.png")
            save_image(fake_e*0.5+0.5, f"outputs/Encrypted/enc_{step}.png")

        del sample
        del fake_d
        del fake_e

        mps.empty_cache()
        gc.collect(generation=2)

    return total_gen_loss/(step+1), total_disc_loss/(step+1)


def training_loop():
    for epoch in range(NUM_EPOCHS):
        gen_D.train(True)
        gen_E.train(True)
        disc_D.train(True)
        disc_E.train(True)

        gen_loss, disc_loss = train_step()

        with torch.no_grad():
            print(f"Epoch: {epoch+1}")
            print(f"Generator Loss: {gen_loss}")
            print(f"Discriminator Loss: {disc_loss}")

            wandb.log({
                "Generator Loss": gen_loss,
                "Discriminator Loss": disc_loss
            })

            if (epoch+1) % 5 == 0:
                torch.save(gen_E.state_dict(),
                           f"Weights/enc_gen/{epoch+1}.pth")
                torch.save(gen_D.state_dict(),
                           f"Weights/dec_gen/{epoch+1}.pth")
                torch.save(disc_E.state_dict(),
                           f"Weights/enc_disc/{epoch+1}.pth")
                torch.save(disc_D.state_dict(),
                           f"Weights/dec_disc/{epoch+1}.pth")


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')

    ids = list(range(1, 1342))

    params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0,
    }

    dataset = ChestXrayDataset(ids)
    wandb.init(
        project='Cryptographic Encryption'
    )
    train_loader = DataLoader(dataset, **params)
    device = torch.device("mps")

    disc_E = Discriminator(in_channels=1).to(device=device)
    disc_D = Discriminator(in_channels=1).to(device=device)
    gen_E = Generator(img_channels=1, num_residuals=3).to(device=device)
    gen_D = Generator(img_channels=1, num_residuals=3).to(device=device)

    NUM_EPOCHS = 100

    disc_opt = torch.optim.Adam(
        list(disc_E.parameters())+list(disc_D.parameters()), lr=1e-5, betas=(0.5, 0.999))
    gen_opt = torch.optim.Adam(
        list(gen_E.parameters())+list(gen_D.parameters()), lr=1e-5, betas=(0.5, 0.999))
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    steps = (len(ids)+params['batch_size']-1)//params['batch_size']

    training_loop()

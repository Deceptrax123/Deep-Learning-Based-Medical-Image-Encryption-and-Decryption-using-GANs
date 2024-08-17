import torch
from torch.utils.data import DataLoader
from chest_xray_dataset import ChestXrayDataset
from torch import nn
from torch import mps
from Models.generator import Generator
from Models.discriminator import Discriminator
import torch.multiprocessing as tmp
import wandb


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')

    ids = list(range(1, 1000))

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }

    dataset = ChestXrayDataset(ids)
    wandb.init(
        project='Cryptographic Encryption'
    )
    train_loader = DataLoader(dataset, **params)
    device = torch.device("mps")

    disc_E = Discriminator().to(device=device)
    disc_D = Discriminator().to(device=device)
    gen_E = Generator(img_channels=3, num_residuals=9).to(device=device)
    gen_D = Generator(img_channels=3, num_residuals=9).to(device=device)

    disc_opt = torch.optim.Adam(
        list(gen_E.parameters())+list(gen_D.parameters()), lr=1e-5, betas=(0.5, 0.999))
    gen_opt = torch.optim.Adam(
        list(disc_E.parameters())+list(disc_D.parameters()), lr=1e-5, betas=(0.5, 0.999))

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

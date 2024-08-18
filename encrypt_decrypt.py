import torch
from torchvision.utils import save_image
from Models.generator import Generator
from Models.encryption import Encryption
from chest_xray_dataset import ChestXrayDataset
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    decrypt = Generator(img_channels=1, num_residuals=3)
    plain = Generator(img_channels=1, num_residuals=3)
    encrypt = Encryption(img_channels=1)

    decrypt.eval()
    encrypt.eval()

    decrypt.load_state_dict(torch.load("Weights/enc_gen/5.pth"), strict=True)
    encrypt.load_state_dict(torch.load("Weights/enc_gen/5.pth"), strict=False)
    plain.load_state_dict(torch.load("Weights/dec_gen/5.pth"), strict=True)

    # Choose a random image to encrypt
    ids = list(range(1, 1342))
    sampled_id = random.choice(ids)

    img = ChestXrayDataset(ids).__getitem__(sampled_id)

    # Encrypt and Decrypt
    img_enc = encrypt(img)
    img_dec = decrypt(img)
    plain_img = plain(img_dec)

    # Visualise using T-SNE
    latent_img = img_enc.detach().numpy()
    latent_img = latent_img.transpose(1, 2, 0)
    latent_comp_3 = TSNE(n_components=3).fit_transform(
        np.reshape(latent_img, (64*64, 256)))

    latent_img_vis = np.reshape(latent_comp_3, (64, 64, 3)).astype(np.uint8)
    latent_img_vis = np.round((latent_img_vis+1)*255)//2
    plt.imshow(latent_img_vis)
    plt.show()

    # Plain and Decrypted Image
    save_image(img_dec*0.5+0.5, "Test_Outputs/Decrypted/output.png")
    save_image(plain_img*0.5+0.5, "Test_Outputs/Plain/output.png")
    save_image(img, "Test_Outputs/Original_Plain/input.png")

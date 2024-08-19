import torch
from torchvision.utils import save_image
from Models.generator import Generator
from Models.encryption import Encryption
from Models.decryption import Decryption
from chest_xray_dataset import ChestXrayDataset
from sklearn.manifold import TSNE
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    generate = Generator(img_channels=1, num_residuals=3)
    encrypt = Encryption(img_channels=1)
    decrypt = Decryption(img_channels=1, num_residuals=3)

    decrypt.eval()
    encrypt.eval()

    generate.load_state_dict(torch.load("Weights/enc_gen/5.pth"), strict=True)
    encrypt.load_state_dict(torch.load("Weights/enc_gen/5.pth"), strict=False)
    decrypt.load_state_dict(torch.load("Weights/enc_gen/5.pth"), strict=False)

    # Choose a random image to encrypt
    ids = list(range(1, 1342))
    sampled_id = random.choice(ids)

    img = ChestXrayDataset(ids).__getitem__(999)

    # Encrypt and Decrypt
    img_enc = encrypt(img)
    img_dec = decrypt(img_enc)
    # Test the cyclic generation of the algorithm
    plain_img = generate(img_dec)

    # Visualise using T-SNE
    latent_img = img_enc.detach().numpy()
    latent_img = latent_img.transpose(1, 2, 0)
    latent_comp_3 = TSNE(n_components=3).fit_transform(
        np.reshape(latent_img, (64*64, 256)))

    latent_img_vis = np.reshape(latent_comp_3, (64, 64, 3)).astype(np.uint8)
    latent_img_vis = np.round((latent_img_vis+1)*255)//2

    # Save Original Image and Decrypted Image
    save_image(img_dec*0.5+0.5, "Test_Outputs/Decrypted/output.png")
    save_image(plain_img*0.5+0.5, "Test_Outputs/Plain/output.png")
    save_image(img, "Test_Outputs/Original_Plain/input.png")

    # Display Outputs on window
    plain = cv2.imread("./Test_Outputs/Plain/output.png")
    decrypted = cv2.imread("./Test_Outputs/Decrypted/output.png")

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(plain)
    ax1.set_title("Input Image")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(latent_img_vis)
    ax2.set_title("Encrypted Image in 2D")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(decrypted)
    ax3.set_title("Decrypted Image")
    plt.show()

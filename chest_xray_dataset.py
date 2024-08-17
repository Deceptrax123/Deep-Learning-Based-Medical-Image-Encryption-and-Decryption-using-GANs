import os
from dotenv import load_dotenv
from PIL import Image
import torchvision
import torch
import torchvision.transforms as T
import numpy as np


class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        load_dotenv('.env')
        data_path = os.getenv("data")

        sample = Image.open(os.path.join(data_path, str(id), ".jpeg"))
        transform_method = T.Compose([
            T.Resize(size=(256, 256)), T.RandomRotation(
                degrees=(-20, +20)), T.TrivialAugmentWide(num_magnitude_bins=31),
            T.ToTensor(), T.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        sample_tensor = transform_method(sample)

        return sample_tensor

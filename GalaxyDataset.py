import torch
import os
from torch.utils.data import Dataset
from skimage import io
import pandas as pd
import numpy as np

class GalaxyDataset(Dataset):
    def __init__(self, csv_file, dir, transform = None):
        super().__init__()

        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(self.data.iloc[index, 0]) + ".jpg")
        image = io.imread(img_path)
        # Figure out y_label
        className = np.argmax(self.data.iloc[index, 1:])
        number = float(className[5:])

        y_label = torch.tensor(number, dtype=float)

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

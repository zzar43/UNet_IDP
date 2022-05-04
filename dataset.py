"""
This file is to create a PyTorch Dataset class for testing purpose.
The data is extracted from the MNIST data.
Two kinds of images: 0 and 1.
Each of image is expanded into a 10 \times 28 \times 28 cube.

Training data: 100
Test data: 20

To create more testing data please check my Google Colab file.

To use:

data_path = os.getcwd() + '/data/3d_MNIST_test/'

training_data = MNIST_3d_test(data_path + 'training_label.csv', data_path + 'training/')
test_data = MNIST_3d_test(data_path + 'test_label.csv', data_path + 'test/')

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)

"""


import os
import pandas as pd

import torch
from torch.utils.data import Dataset

class MNIST_3d_test(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torch.load(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


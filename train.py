import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataset import MNIST_3d_test
from model import UNet_3D_with_DS

import os
import pandas as pd
from tqdm import tqdm

# hyperparameters
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 5
epochs = 5
data_path = 'data/3d_MNIST_test/'
out_label_num = 2

# train and test function
def train(dataloader, model, loss_fn, optimizer):
    loop = tqdm(dataloader)
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# main
def main():
    # dataset
    training_data = MNIST_3d_test(data_path + 'training_label.csv', data_path + 'training/')
    test_data = MNIST_3d_test(data_path + 'test_label.csv', data_path + 'test/')
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # train setup
    model = UNet_3D_with_DS(in_channels=1, out_num=out_label_num, features_down=[4,16,32], features_up=[16,8]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    torch.save(model, 'model/')
    print("Done!")

if __name__ == "__main__":
    main()
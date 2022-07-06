import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import numpy as np

from dataset import MNIST_3d_test
from model import UNet_3D_with_DS

import os
import pandas as pd
from tqdm import tqdm

# hyperparameters
NEW_TRAINING = True
DATA_PATH = "/export/home/da.li1/dataset/UNet_IDP/UNet_IDP_Stage_1"
LABEL_FILE = "UNet_IDP_Stage_1.csv"
# random seed
torch.manual_seed(42)
np.random.seed(42)
# configuration
k_folds = 5
learning_rate = 1e-3
device = "cuda:1" if torch.cuda.is_available() else "cpu"
batch_size = 40
epochs = 20
out_label_num = 4

# train and test function
def train(dataloader, model, loss_fn, optimizer):
    loop = tqdm(dataloader, ncols=80)
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
    # k-fold
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # dataset
    dataset = MNIST_3d_test(os.path.join(DATA_PATH, LABEL_FILE), os.path.join(DATA_PATH, "Dataset"))
    # build dataset for one portion
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # sample
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        # build data loader
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)
        break

    print('Training data:\n', train_ids)
    print('Test data:\n', test_ids)

    # train setup
    if NEW_TRAINING == True:
        model = UNet_3D_with_DS(in_channels=1, out_num=out_label_num, features_down=[4,16,32,64], features_up=[32,16,8]).to(device)

        if torch.cuda.is_available():
            model.cuda()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    else:
        path = os.getcwd()
        model = torch.load(path+'/model/model.pt').to(device)
        print("Model loaded from: ", path+'/model/model.pt')
    
    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    path = os.getcwd()
    torch.save(model, path+'/model/model.pt')
    print("Model is saved to: ", path+'/model/model.pt')
    print("Done!")

if __name__ == "__main__":
    main()
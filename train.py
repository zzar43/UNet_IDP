import torch
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

from dataset import MNIST_3d_test
from model import UNet_3D_with_DS

import os
import numpy as np
from tqdm import tqdm
import time
from sklearn.model_selection import KFold

NEW_TRAINING = True
K_FOLD = True
DATA_PATH = "/export/home/da.li1/dataset/UNet_IDP/UNet_IDP_Stage_1"
LABEL_FILE = "UNet_IDP_Stage_1.csv"
SAVE_NAME = "20220704_1"

# hyperparameters
learning_rate = 1e-2
# learning_rate = 1e-3
# learning_rate = 5e-4
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 40
epochs = 200
data_path = "/export/home/da.li1/dataset/UNet_IDP/UNet_IDP_Stage_1"
out_label_num = 4
k_folds = 5

# train and test function
def train(dataloader, model, loss_fn, optimizer):
    loop = tqdm(dataloader, ncols=80)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    for batch, (X, y) in enumerate(loop):
    # for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # update loss
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    train_loss /= num_batches
    return train_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # model.eval()
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
    return test_loss, 100*correct

# main
def main():
    # record loss and accuracy
    train_loss_record = []
    test_loss_record = []
    accuracy_record = []

    # set time
    tic = time.time()
    # print basic information
    print("Learning rate: ", learning_rate)
    print("Batch size: ", batch_size)
    print("Total epochs: ", epochs)

    if K_FOLD == True:
        print("K-Fold is used with k = ", k_folds)
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
    else:
        training_data = MNIST_3d_test(DATA_PATH + '/UNet_IDP_Stage_1.csv', DATA_PATH + '/Dataset')
        test_data = MNIST_3d_test(DATA_PATH + '/UNet_IDP_Stage_1.csv', DATA_PATH + '/Dataset')
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # train setup
    if NEW_TRAINING == True:
        model = UNet_3D_with_DS(in_channels=1, out_num=out_label_num, features_down=[4,16,32,64], features_up=[32,16,8], dropout_p=0).to(device)

        if torch.cuda.is_available():
            model.cuda()
        if torch.cuda.device_count() > 1:
            print("Data parallel: ", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    else:
        path = os.getcwd()
        model = UNet_3D_with_DS(in_channels=1, out_num=out_label_num, features_down=[4,16,32,64], features_up=[32,16,8], dropout_p=0).to(device)
        model.load_state_dict(torch.load(path+'/model/model_20220624_3.pt'))
        print("Model loaded from: ", path+'/model/model_20220624_3.pt')

        if torch.cuda.is_available():
            model.cuda()
        if torch.cuda.device_count() > 1:
            print("Data parallel: ", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    
    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # main train loop
    if K_FOLD == True:
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train(train_dataloader, model, loss_fn, optimizer)
            train_loss_record.append(train_loss)
            test_loss, accuracy = test(test_dataloader, model, loss_fn)
            test_loss_record.append(test_loss)
            accuracy_record.append(accuracy)
    else:
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train(train_dataloader, model, loss_fn, optimizer)
            train_loss_record.append(train_loss)
        test_loss, accuracy = test(test_dataloader, model, loss_fn)
        test_loss_record.append(test_loss)
        accuracy_record.append(accuracy)

    # compute elapsed time
    toc = time.time()
    print("Elapsed time is: ", (toc-tic)/60, " min.")

    # save model
    save_model_name = "model_" + SAVE_NAME + ".pt"
    torch.save(model.module.state_dict(), "model/"+save_model_name)
    save_data_name = "data" + SAVE_NAME + ".npy"
    np.save(save_data_name, [train_loss_record, test_loss_record, accuracy_record])

    print("Model is saved to: ", "model/"+save_model_name)
    print("Data is saved as: ", save_data_name)
    print("Done!")

if __name__ == "__main__":
    main()
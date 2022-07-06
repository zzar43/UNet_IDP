import torch
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

from dataset import MNIST_3d_test
from model import UNet_3D_with_DS

import os
# import pandas as pd
# from tqdm import tqdm
import time

# hyperparameters
data_path = "/export/home/da.li1/dataset/UNet_IDP/UNet_IDP_Stage_1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
out_label_num = 4
batch_size = 40

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


# main
def main():
    # set time 
    tic = time.time()

    test_data = MNIST_3d_test(data_path + '/UNet_IDP_Stage_1.csv', data_path + '/Dataset')
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # load model
    path = os.getcwd()
    model = UNet_3D_with_DS(in_channels=1, out_num=out_label_num, features_down=[4,16,32,64], features_up=[32,16,8], dropout_p=0).to(device)
    model.load_state_dict(torch.load(path+'/model/model_20220627.pt'))
    print("Model loaded from: ", path+'/model/model_20220627.pt')

    # data parallel
    if torch.cuda.is_available():
            model.cuda()
    if torch.cuda.device_count() > 1:
        print("Data parallel: ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    test(test_dataloader, model, loss_fn)

    # compute elapsed time
    toc = time.time()
    print("Elapsed time is: ", (toc-tic)/60, " min.")

if __name__ == "__main__":
    main()



    
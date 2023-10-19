import numpy as np
from utils import one_hot_encoding, gen_cube, MyDataset
from torch.utils.data import DataLoader
import torch

def transform_data4ResNet(X_test,y_test, X_train=None,y_train=None,device="cpu",batch_s=(64,64)):

    # TODO fix for both methods batch_s=None, i.e. can i do everything using loaders?

    # first transform the test set and both labels
    test_set_cube = np.array([gen_cube(acl) for acl in X_test.tolist()])

    # case in which I just need the grouping #TODO is it really necessary??
    if X_train is None and y_train is None and y_test is None:
        return torch.tensor(test_set_cube).type(torch.int64).to(device)

    y_train,y_test,enc = one_hot_encoding( y_train,y_test )
    X_test,y_test = torch.tensor(test_set_cube).type(torch.float), torch.tensor(y_test).type(torch.int64)

    # in case X_train is provided, do the previous steps also for the training set
    if not(X_train is None):
        train_set_cube = np.array([gen_cube(acl) for acl in X_train.tolist()])
        X_train,y_train = torch.tensor(train_set_cube).type(torch.float), torch.tensor(y_train).type(torch.int64)
    else:
        train_loader = None

    if batch_s is None:
        # if batch_s is not provided return tensors not loaders #TODO is it necessary???
        return  X_train, y_train,X_test,y_test, enc
    else:
        # otherwise return the loaders
        train_loader = DataLoader(MyDataset(X_train,y_train), batch_size=batch_s[0],shuffle=True)
        test_loader = DataLoader(MyDataset(X_test,y_test), batch_size=batch_s[1],shuffle=False)
        return train_loader, test_loader, enc


def transform2tensors(X_train, y_train, X_test,y_test,batch_size=None,device="cpu"):
    X_train = torch.tensor( X_train  ).type(torch.float).to(device)
    X_test = torch.tensor( X_test  ).type(torch.float).to(device)


    y_train,y_test,enc = one_hot_encoding( y_train,y_test )
    y_train = torch.Tensor( y_train ).type(torch.int64).to(device)
    y_test = torch.Tensor( y_test ).type(torch.int64).to(device)

    if batch_size==None:

        return X_train, y_train, X_test, y_test, enc

    else:
        train_loader = DataLoader(MyDataset(X_train,y_train), batch_size=batch_size[0],shuffle=True)
        test_loader = DataLoader(MyDataset(X_test,y_test), batch_size=batch_size[1],shuffle=False)
        return train_loader, test_loader, enc
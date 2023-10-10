import numpy as np
from utils import one_hot_encoding, gen_cube, MyDataset
from torch.utils.data import DataLoader
import torch

def transform_data4ResNet(X_train, y_train, X_test,y_test,device="cpu",batch_s=(64,64)):

    #TODO option for only test (put everything into a for loop)

    train_set_cube = np.array([gen_cube(acl) for acl in X_train.tolist()])
    test_set_cube = np.array([gen_cube(acl) for acl in X_test.tolist()])

    y_train,y_test,enc = one_hot_encoding( y_train,y_test )

    X_train,y_train = torch.tensor(train_set_cube).type(torch.float), torch.tensor(y_train).type(torch.int64)
    X_test,y_test = torch.tensor(test_set_cube).type(torch.float), torch.tensor(y_test).type(torch.int64)

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
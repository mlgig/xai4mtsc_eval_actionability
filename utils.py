import torch
from torch.cuda import  is_available as is_gpu_available
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dCAM.src.models.CNN_models import TSDataset
from torch.utils import data


def gen_cube(instance):
    result = []
    for i in range(len(instance)):
        result.append([instance[(i+j)%len(instance)] for j in range(len(instance))])
    return result

def one_hot_encoding(train_labels,test_labels):
    enc = LabelEncoder()
    y_train = enc.fit_transform(train_labels)
    y_test = enc.transform(test_labels)

    return y_train,y_test,enc


#TODO move into pytorch_utils.py?

def transform_data4ResNet(X_train, y_train, X_test,y_test,gen_cube=False,device="cpu"):

    #TODO option for only test (put everything into a foor loop)
    # get dataset loaders
    n_channels =X_train.shape[1]# if not concat else 1
    n_classes = len( np.unique(y_train) )

    if gen_cube:
        train_set_cube = np.array([gen_cube(acl) for acl in X_train.tolist()])
        test_set_cube = np.array([gen_cube(acl) for acl in X_test.tolist()])
        X_train = train_set_cube
        X_test = test_set_cube

    #TODO bringing it back
    batch_s = (64,64)

    y_train,y_test,enc = one_hot_encoding( y_train,y_test )

    X_train,y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_test,y_test = torch.tensor(X_test), torch.tensor(y_test)

    train_loader = DataLoader(MyDataset(X_train,y_train), batch_size=batch_s[0],shuffle=True)
    test_loader = DataLoader(MyDataset(X_test,y_test), batch_size=batch_s[1],shuffle=False)

    return train_loader, test_loader,n_channels,n_classes, device, enc

def pre_fature_normalization(X_train,X_test):
    eps = 1e-6
    f_mean = X_train.mean(axis=0, keepdims=True)
    f_std = X_train.std(axis=0, keepdims=True) + eps  # epsilon to avoid dividing by 0
    X_train_tfm2 = (X_train - f_mean) / f_std
    X_test_tfm2 = (X_test - f_mean) / f_std
    return  X_train_tfm2,X_test_tfm2


class MyDataset(data.Dataset):

    def __init__(self,X,y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
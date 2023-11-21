import numpy as np
from utils import one_hot_encoding, gen_cube, MyDataset
from torch.utils.data import DataLoader
import torch

def transform_data4ResNet( X_train,y_train,X_test,y_test,device="cpu",batch_s=(64,64)):

    # TODO fix for both methods batch_s=None, i.e. can i do everything using loaders?

    # first transform labels
    y_train,y_test,enc = one_hot_encoding( y_train,y_test )

    # then transform train and test set
    train_set_cube = np.array([gen_cube(acl) for acl in X_train.tolist()])
    X_train = torch.tensor(train_set_cube).type(torch.float)
    y_train = torch.tensor(y_train).type(torch.int64)

    test_set_cube = np.array( [gen_cube(acl) for acl in X_test.tolist()] )
    X_test = torch.tensor(test_set_cube).type(torch.float)
    y_test =  torch.tensor(y_test).type(torch.int64)

    # return loader or single tensors accordingly to batch_s
    if batch_s is None:
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


##### ConvTrans ####



class dataset_class(torch.utils.data.Dataset):

    def __init__(self, data, label):
        super(dataset_class, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, ind):

        x = self.feature[ind]
        x = x.astype(np.float32)

        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y).type(torch.LongTensor)

        return data, label, ind

    def __len__(self):
        return len(self.labels)


def transform4ConvTran( train_X, train_y,test_X, test_y, hyper_params, n_classes, batch_s):
    train_y, test_y, enc = one_hot_encoding(train_y, test_y)
    train_dataset = dataset_class(train_X, train_y)
    test_dataset = dataset_class(test_X, test_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_s[0], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_s[1], shuffle=False, pin_memory=True)
    hyper_params['num_labels'] = n_classes
    hyper_params['Data_shape'] = train_loader.dataset.feature.shape
    hyper_params['loss_module'] = torch.nn.CrossEntropyLoss(reduction='none')
    return train_loader,test_loader, hyper_params

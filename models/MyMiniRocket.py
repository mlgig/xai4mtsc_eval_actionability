from torch.cuda import empty_cache as empty_gpu_cache
import torch.nn as nn
from models.tsai.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features
from models.LogisticRegression import LogisticRegression
import numpy as np
from utils import pre_fature_normalization, MyDataset
from torch.utils.data import  DataLoader
import torch

class MyMiniRocket(nn.Module):

    def __init__(self, n_channels,seq_len,n_classes, device="cuda"):
        super(MyMiniRocket, self).__init__()

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.transformer = MiniRocketFeatures(n_channels,seq_len,device=device)
        self.classifier = LogisticRegression(9996,n_classes,nn.CrossEntropyLoss())

    def transform_dataset(self,X_train,X_test,chunksize):

        self.transformer.fit(X_train)

        X_train_trans = get_minirocket_features(X_train,self.transformer, chunksize=chunksize,to_np=False)
        empty_gpu_cache()
        X_test_trans = get_minirocket_features(X_test, self.transformer, chunksize=chunksize,to_np=False)
        empty_gpu_cache()

        X_train_trans_norm, X_test_trans_norm = pre_fature_normalization(X_train_trans,X_test_trans)

        return X_train_trans_norm, X_test_trans_norm

    def train_regression(self, X_train, y_train,X_test,y_test, Cs, k, batch_size):

        # TODO parameter!
        # K-fold cross validation to find the best C parameter
        n_classes = 2 #len(np.unique(y_train))
        self.classifier = LogisticRegression(9996,n_classes,nn.CrossEntropyLoss())
        best_loss ,best_C = self.classifier.validation(Cs,k,X_train,y_train,batch_size)

        # final train using the hole train set based on the C previously found
        # TODO batch size as param!
        train_loader = DataLoader( MyDataset(X_train,y_train), batch_size=64,  shuffle=True)
        test_loader = DataLoader( MyDataset(X_test,y_test), batch_size=50000,  shuffle=False)

        self.classifier = LogisticRegression(9996,n_classes,nn.CrossEntropyLoss())
        # TODO param for lr"
        self.classifier.optimizer = torch.optim.Adam(self.parameters(), lr= 0.0001,weight_decay=1/best_C)
        self.classifier.final_train(train_loader,test_loader)

from torch.cuda import empty_cache as empty_gpu_cache
import torch.nn as nn
from models.tsai.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features
from models.LogisticRegression import LogisticRegression
import numpy as np
from utils import pre_fature_normalization, MyDataset
from torch.utils.data import  DataLoader
import torch
from models.tsai.ROCKET_Pytorch import ROCKET

class MyMiniRocket(nn.Module):

    def __init__(self, transformer, n_channels,seq_len,n_classes, device="cuda"):
        super(MyMiniRocket, self).__init__()

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.n_classes = n_classes
        if transformer=="MiniRocket":
            self.transformer_model = MiniRocketFeatures(n_channels,seq_len,device=device)
            self.intermediate_dim = 9996
        elif transformer=="Rocket":
            self.transformer_model = ROCKET(c_in=n_channels,seq_len=seq_len)
            self.intermediate_dim = 20000
        else:
            raise ValueError("transformer can be either MiniRocket or Rocket")
        self.classifier = LogisticRegression(self.intermediate_dim,n_classes,nn.CrossEntropyLoss())      #20 000 intermediate dim as class instance

    def forward(self,X):
        X_trans = self.transformer_model(X)
        y = self.classifier(X_trans)
        probs = nn.functional.softmax( y, dim=-1 )
        return probs

    def transform_dataset(self,X_train,X_test,chunksize):

        # TODO need to be set up for rocket
        self.transformer_model.fit(X_train)

        X_train_trans = get_minirocket_features(X_train,self.transformer_model, chunksize=chunksize,to_np=False)
        empty_gpu_cache()
        X_test_trans = get_minirocket_features(X_test, self.transformer_model, chunksize=chunksize,to_np=False)
        empty_gpu_cache()

        X_train_trans_norm, X_test_trans_norm = pre_fature_normalization(X_train_trans,X_test_trans)

        return X_train_trans_norm, X_test_trans_norm

    def train_regression(self, X_train, y_train,X_test,y_test, Cs, k, batch_size):

        # K-fold cross validation to find the best C parameter
        n_classes = len(torch.unique(y_train))
        self.classifier = LogisticRegression(self.intermediate_dim,n_classes,nn.CrossEntropyLoss())
        best_loss ,best_C = self.classifier.validation(Cs,k,X_train,y_train,batch_size)

        # final train using the hole train set based on the C previously found
        train_loader = DataLoader( MyDataset(X_train,y_train), batch_size=batch_size,  shuffle=True)
        test_loader = DataLoader( MyDataset(X_test,y_test), batch_size=X_test.shape[0],  shuffle=False)

        # TODO substitue 9996 with self.intermediate dim
        self.classifier = LogisticRegression(self.intermediate_dim,n_classes,nn.CrossEntropyLoss())
        self.classifier.optimizer = torch.optim.Adam(self.parameters(), lr=self.classifier.learning_rate ,weight_decay=1/best_C)
        self.classifier.final_train(train_loader,test_loader)

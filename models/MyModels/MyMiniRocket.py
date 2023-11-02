from torch.cuda import empty_cache as empty_gpu_cache
import torch.nn as nn
from models.tsai.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features
from models.MyModels.LogisticRegression import LogisticRegression
import timeit
from utils import MyDataset
from torch.utils.data import  DataLoader
import torch
from models.tsai.ROCKET_Pytorch import ROCKET
import numpy as np

# TODO should I change module and file name??
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
            self.transformer_model = ROCKET(n_channels,seq_len, device=device)
            self.intermediate_dim = 20000
        else:
            raise ValueError("transformer can be either MiniRocket or Rocket")
        self.classifier = LogisticRegression(self.intermediate_dim, n_classes, nn.CrossEntropyLoss())
        self.to(device)

    def forward(self,X):
        X_trans,_ = self.transform_dataset(X,X,normalise=True,chunk_size=512)
        y = self.classifier(X_trans)
        return y

    def trainAndScore(self,X_train,y_train,X_test,y_test, normalise=True, chunk_size=512):
        start = timeit.default_timer()
        X_train_trans, X_test_trans = self.transform_dataset(X_train,X_test,chunk_size=chunk_size, normalise=normalise)
        print("transformation done in ", timeit.default_timer() - start)
        torch.cuda.empty_cache()

        start = timeit.default_timer()
        Cs = np.logspace(-1,1,3) #np.logspace(-4,4,10)
        acc =  self.train_regression(X_train_trans, y_train, X_test_trans, y_test,Cs =Cs,k=5,batch_size=64 )
        print("classifier in ", timeit.default_timer() - start, " accuracy of ",
              type(self.transformer_model)," was ", acc)
        return acc

    def transform_dataset(self,X_train,X_test,chunk_size,normalise):

        def transform_miniRocket(X_train,X_test, chunk_size,normalise):

            self.transformer_model.fit(X_train[:200])
            # transforming features
            X_train_trans = get_minirocket_features(X_train, self.transformer_model, chunksize=chunk_size, to_np=False)
            empty_gpu_cache()
            X_test_trans = get_minirocket_features(X_test, self.transformer_model, chunksize=chunk_size, to_np=False)
            empty_gpu_cache()

            if normalise:
                X_train, X_test =  self.__pre_fature_normalization(X_train_trans, X_test_trans)
            else:
                X_train, X_test =  X_train_trans, X_test_trans

            return X_train.squeeze() , X_test.squeeze()

        def transform_Rocket(X_train,X_test,normalise):

            # transform features
            X_train_trans = self.transformer_model(X_train)
            empty_gpu_cache()
            X_test_trans = self.transformer_model(X_test)
            empty_gpu_cache()

            if normalise:
                X_train, X_test =  self.__pre_fature_normalization(X_train_trans, X_test_trans)
            else:
                X_train, X_test =  X_train_trans, X_test_trans

            return X_train , X_test


        if isinstance(self.transformer_model,MiniRocketFeatures):
            return transform_miniRocket(X_train, X_test, chunk_size,normalise)
        else:
            return transform_Rocket(X_train,X_test,normalise)


    def train_regression(self, X_train, y_train,X_test,y_test, Cs, k, batch_size):

        # K-fold cross validation to find the best C parameter
        n_classes = len(torch.unique(y_train))
        best_loss ,best_C = self.classifier.validation(Cs,k,X_train,y_train,batch_size)

        # final train using the hole train set based on the C previously found
        train_loader = DataLoader( MyDataset(X_train,y_train), batch_size=batch_size,  shuffle=True)
        test_loader = DataLoader( MyDataset(X_test,y_test), batch_size=X_test.shape[0],  shuffle=False)
        self.classifier = LogisticRegression(self.intermediate_dim,n_classes,nn.CrossEntropyLoss())
        self.classifier.optimizer = torch.optim.Adam(self.parameters(), lr=self.classifier.learning_rate ,weight_decay=1/best_C)
        accuracy = self.classifier.final_train(train_loader,test_loader)

        # and return the found accuracy
        return accuracy

    def __pre_fature_normalization(self,X_train,X_test):
        eps = 1e-6
        f_mean = X_train.mean(axis=0, keepdims=True)
        f_std = X_train.std(axis=0, keepdims=True) + eps  # epsilon to avoid dividing by 0
        X_train_tfm2 = (X_train - f_mean) / f_std
        X_test_tfm2 = (X_test - f_mean) / f_std
        return  X_train_tfm2,X_test_tfm2
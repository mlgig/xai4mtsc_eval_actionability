import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from utils import MyDataset
from torch.utils.data import DataLoader

class LogisticRegression(torch.nn.Module):

    # TODO device to be set CPU and move the cuda parameter setting in main
    def __init__(self, input_dim, output_dim,criterion,learning_rate= 0.0001,device="cuda"):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.outpit_dim = output_dim
        self.linear = torch.nn.Linear(self.input_dim, self.outpit_dim,bias=True)
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.device=device

        # TODO following 3 need to be pass as parameters in the constructor
        self.max_epoch = 300
        self.check_every = 20
        self.early_stop_after = 60

        self.no_improving_steps = 0
        self.best_accuracy = 0.0
        self.best_loss = 0.0
        self.best_epoch = 0
        # TODO can I do better here?
        self.optimizer = None #torch.optim.Adam(self.parameters(), lr= learning_rate,weight_decay=1/C)

        self.to(device)


    def __train_step__(self,train_loader):
        losses = []
        for i,batch_data_train in enumerate(train_loader):
            X,y = batch_data_train
            self.optimizer.zero_grad()
            output = self(X)
            loss = self.criterion(output, y)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            self.optimizer.step()
        return  np.average( np.array(losses) )

    def __test_accuracy__(self,val_loader):
        with torch.no_grad():
            for i,batch_data_val in enumerate(val_loader):
                X_test, y_test = batch_data_val
                output_test = self(X_test)
                loss_test = self.criterion(output_test,y_test)
                predicted_test = torch.max(output_test, axis=1)
                test_acc = accuracy_score(y_test.cpu(), predicted_test.indices.cpu())

                if test_acc>self.best_accuracy:
                    self.best_accuracy, self.best_loss = test_acc , loss_test.cpu().numpy()
                else:
                    self.no_improving_steps+=self.check_every


            return self.best_accuracy, self.no_improving_steps>= self.early_stop_after


    def __reset_model__(self, C):
        self.linear = torch.nn.Linear(self.input_dim, self.outpit_dim,bias=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1 / C)
        self.best_accuracy = 0.0
        self.best_loss = 0.0
        self.best_epoch = 0
        self.no_improving_steps = 0

    def __Kfold_split__(self,X,y,k,i, batch_size):
        # single fold length
        tot_len = X.shape[0]
        val_len = tot_len/k

        # split data according to the i-th fold
        val_X = X[  int(i*val_len): int( (i+1)*val_len) ]
        val_y = y[  int( i*val_len ): int( (i+1)*val_len ) ]
        val_loader = DataLoader( MyDataset(val_X,val_y), batch_size=50000,  shuffle=False)

        train_X = torch.concat( (
            X [ :int( i*val_len )],
            X [ int( (i+1)*val_len ): ]
        ))
        train_y = torch.concat( (
            y [ :int( i*val_len )],
            y [ int( (i+1)*val_len ): ]
        ))
        train_loader = DataLoader( MyDataset(train_X,train_y), batch_size=batch_size,  shuffle=True)

        return train_loader, val_loader


    def forward(self, x):
        # TODO softmax for more than 2 classes
        outputs = torch.sigmoid(self.linear(x))
        #torch.sum( torch.softmax(outputs, dim=1), dim=1 )
        return outputs

    def validation(self,Cs, k, X_train, y_train, batch_size):
        # store results
        scores = np.zeros( (Cs.shape[0],k) )

        # splits data ino k folds
        for i in range(k):
            train_loader, val_loader = self.__Kfold_split__(X_train,y_train,k=k,i=i,batch_size=batch_size)

            for j,C in enumerate(Cs):
                in_dim = X_train.shape[1]
                #TODO pass as parameter!
                n_classes = 2

                #reset the model
                self.__reset_model__(C)
                #TODO device as parameter!

                for epoch in range(self.max_epoch):
                    loss = self.__train_step__(train_loader)

                    if epoch % self.check_every ==0:
                        accuracy, to_stop = self.__test_accuracy__(val_loader)
                        print("epoch", epoch, " accuracy ", accuracy, self.best_accuracy)
                        if to_stop:
                            print("EAERLY STOPPED AT", epoch, "\n\n")
                            scores[j,i]+=[accuracy]
                            break

        avg_scores = np.average( scores, axis=-1)
        print("cv results:",avg_scores)
        best_idx  =np.argmax( avg_scores )
        best_param = Cs[best_idx]
        return avg_scores[best_idx], best_param


    def final_train(self,train_loader,test_loader):

        for epoch in range(self.max_epoch):
            self.__train_step__(train_loader)

            if epoch%self.check_every==0:

                accuracy, to_stop = self.__test_accuracy__(test_loader)
                if to_stop:
                    print("final accuracy", accuracy, self.best_accuracy, "epoch", epoch)
                    break

from models.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN
from pytorch_utils import transform_data4ResNet, transform2tensors
from models.MyModels.MyMiniRocket import MyMiniRocket
from load_data import load_data
import torch
import os
import numpy as np
import timeit
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from joblib import dump

def main():


    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset_name in ['synth_1line','synth_2lines']:
        train_X, train_y, test_X, test_y, seq_len, n_channels, n_classes = load_data(dataset_name)


        for n in range(5):
            
            for transformer in ["MiniRocket"]:#, "Rocket","MiniRocket"]:

                X_train,y_train,X_test,y_test, enc = transform2tensors(train_X,train_y,test_X,test_y,device=device)
                miniRocket = MyMiniRocket(transformer,n_channels,seq_len,n_classes,normalise=True,verbose=False,
                                          device=device)
                acc = miniRocket.trainAndScore(X_train,y_train,X_test,y_test)

                model_name = "_".join( (transformer,"normalTrue", str(n), str(acc)[2:5] ) )
                file_path= "//".join( ("saved_models",dataset_name,model_name) )
                torch.save(miniRocket,file_path+".pt")
                torch.cuda.empty_cache()


        # TODO check batch sizes and organise them accordingly to the dataset !
        # dResNet

        train_loader, test_loader, enc= transform_data4ResNet(X_train= train_X, y_train= train_y,
                X_test= test_X, y_test= test_y,device=device, batch_s=(32,32))

        for n in range(5):
            for n_filters in [64,128]:
                resNet = dResNetBaseline(n_channels, mid_channels=n_filters,num_pred_classes= n_classes).to(device)
                model = ModelCNN(model=resNet, n_epochs_stop=50,device=device, save_path="saved_models/tmp.pt")
                acc = model.train(num_epochs=301,train_loader=train_loader,test_loader=test_loader)
                model_name = "_".join( ("dResNet", str(n_filters), str(n), str(acc)[2:5] ) )
                file_path= "/".join( ("saved_models",dataset_name,model_name) )
                print(file_path)
                #os.rename("saved_models/tmp.pt",file_path+".pt") #torch.save(resNet, file_path+".pt")
                torch.cuda.empty_cache()
                print("dResNet accuracy was ",acc)


        # ResNet
        train_loader, test_loader,enc = (
            transform2tensors(train_X,train_y,test_X,test_y, batch_size=(128,128), device=device ))

        for n in range(5):
            for n_filters in [64,128]:
                resNet = ResNetBaseline(n_channels, mid_channels=n_filters,num_pred_classes= n_classes).to(device)
                model = ModelCNN(model=resNet, n_epochs_stop=50,device=device, save_path="saved_models/tmp.pt")
                acc = model.train(num_epochs=301,train_loader=train_loader,test_loader=test_loader)
                model_name = "_".join( ("ResNet", str(n_filters), str(n), str(acc)[2:5] ) )
                file_path= "/".join( ("saved_models",dataset_name,model_name) )
                print(file_path)
                #os.rename("saved_models/tmp.pt",file_path+".pt") #torch.save(resNet, file_path+".pt")
                torch.cuda.empty_cache()
                print("resNet accuracy was ",acc)


        exit()
if __name__ == "__main__" :
    main()
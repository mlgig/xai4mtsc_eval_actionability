from models.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN
from pytorch_utils import transform_data4ResNet, transform2tensors
from models.MyModels.MyMiniRocket import MyMiniRocket
from load_data import load_data
from utils import MyDataset
import torch
import numpy as np
import timeit
from sklearn.linear_model import RidgeClassifierCV
from joblib import dump

def main():


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO put back the correct datasets order
    # TODO avoid a lot of prints in the training stage
    for dataset_name in ['MP', 'synth_2lines', 'CMJ','synth_1line', ]:
        train_X, train_y,test_X,test_y, seq_len, n_channels, n_classes = load_data(dataset_name)

        for n in range(1):

            for transformer in ["MiniRocket", "Rocket"]:

                # TODO implement a method in MyModel taking care of both transforming dataset and
                # training the linear regression
                X_train,y_train,X_test,y_test, enc = transform2tensors(train_X,train_y,test_X,test_y,device=device)
                miniRocket = MyMiniRocket(transformer,n_channels,seq_len,n_classes)

                start = timeit.default_timer()
                X_train_trans, X_test_trans = miniRocket.transform_dataset(X_train=X_train,X_test=X_test, chunk_size=512,normalise=True)
                print("trans in ", timeit.default_timer() - start)
                torch.cuda.empty_cache()

                start = timeit.default_timer()
                Cs = np.logspace(-4,4,10)
                acc = miniRocket.train_regression( X_train_trans, y_train, X_test_trans, y_test,Cs =Cs,k=5,batch_size=128 )
                print("classifier in ", timeit.default_timer() - start, transformer, " accuracy was ", acc)

                model_name = "_".join( (transformer, str(n), str(acc)[2:5] ) )
                file_path= "//".join( ("saved_models",dataset_name,model_name) )
                torch.save(miniRocket,file_path+".pt")


            # TODO check batch sizes and organise them accordingly to the dataset !
            # dResNet
            for n_filters in [64,128]:
                train_loader, test_loader, enc= (
                    transform_data4ResNet(train_X,train_y,test_X,test_y,device=device))
                resNet = dResNetBaseline(n_channels, mid_channels=n_filters,num_pred_classes= n_classes).to(device)
                model = ModelCNN(model=resNet, n_epochs_stop=30,device=device)
                acc = model.train(num_epochs=200,train_loader=train_loader,test_loader=test_loader)
                model_name = "_".join( ("dResNet", str(n_filters), str(n), str(acc)[2:5] ) )
                file_path= "//".join( ("saved_models",dataset_name,model_name) )
                torch.save(resNet, file_path+".pt")
                print("dResNet accuracy was ",acc)


            # dResNet
            for n_filters in [64,128]:
                train_loader, test_loader,enc = (
                    transform2tensors(train_X,train_y,test_X,test_y, batch_size=(64,64), device=device ))
                resNet = ResNetBaseline(n_channels, mid_channels=n_filters,num_pred_classes= n_classes).to(device)
                model = ModelCNN(model=resNet, n_epochs_stop=30,device=device)
                acc = model.train(num_epochs=200,train_loader=train_loader,test_loader=test_loader)
                model_name = "_".join( ("ResNet", str(n_filters), str(n), str(acc)[2:4] ) )
                file_path= "//".join( ("saved_models",dataset_name,model_name) )
                torch.save(resNet, file_path+".pt")
                print("resNet accuracy was ",acc)

if __name__ == "__main__" :
    main()
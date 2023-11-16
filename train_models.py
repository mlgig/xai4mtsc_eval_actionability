from models.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN
from pytorch_utils import transform_data4ResNet, transform2tensors
from models.MyModels.MyMiniRocket import MyMiniRocket
from load_data import load_data
import torch
import os

def main():


    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset_name in ['synth_1line', 'synth_2lines']:
        train_X, train_y, test_X, test_y, seq_len, n_channels, n_classes = load_data(dataset_name)
        transform_data_dResnet = lambda : transform_data4ResNet(X_train= train_X, y_train= train_y,
                                             X_test= test_X, y_test= test_y,device=device, batch_s=(128,128))
        transform_data_resnet = lambda : transform2tensors(train_X,train_y,test_X,test_y, batch_size=(128,128), device=device )


        for n in range(1):
            # dResNet / resMet

            for model_n in ["resNet","dResNet"]:
                for n_filters in [64]:

                    if model_n=="resNet":
                        train_loader, test_loader, enc= transform_data_resnet()
                        resNet = ResNetBaseline(n_channels, mid_channels=n_filters,num_pred_classes= n_classes).to(device)
                    else:
                        train_loader, test_loader, enc= transform_data_dResnet()
                        resNet = dResNetBaseline(n_channels, mid_channels=n_filters,num_pred_classes= n_classes).to(device)

                    model = ModelCNN(model=resNet, n_epochs_stop=50,device=device, save_path="saved_models/tmp.pt")
                    acc = model.train(num_epochs=11,train_loader=train_loader,test_loader=test_loader)

                    model_name = "_".join( (model_n, str(n_filters), str(n), str(acc)[2:5] ) )
                    file_path= "/".join( ("saved_models",dataset_name,model_name) )
                    os.rename("saved_models/tmp.pt",file_path+".pt") #torch.save(resNet, file_path+".pt")
                    torch.cuda.empty_cache()
                    print(model_n," accuracy was ",acc)

            # Rocket/ MiniRocket
            for transformer in [ "MiniRocket", "Rocket"]:

                X_train,y_train,X_test,y_test, enc = transform2tensors(train_X,train_y,test_X,test_y,device=device)
                miniRocket = MyMiniRocket(transformer,n_channels,seq_len,n_classes,normalise=True,verbose=False,
                                          device=device)
                acc = miniRocket.trainAndScore(X_train,y_train,X_test,y_test)

                model_name = "_".join( (transformer,"normalTrue", str(n), str(acc)[2:5] ) )
                file_path= "//".join( ("saved_models",dataset_name,model_name) )
                torch.save(miniRocket,file_path+".pt")
                torch.cuda.empty_cache()

if __name__ == "__main__" :
    main()
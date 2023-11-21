from models.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN, InceptionModel, dInceptionModel
from pytorch_utils import transform_data4ResNet, transform2tensors, transform4ConvTran
from models.MyModels.MyMiniRocket import MyMiniRocket
from load_data import load_data
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from models.ConvTran.Models.utils import load_model as load_transformer
from models.ConvTran.utils import Initialization
from models.ConvTran.Models.model import ConvTran
import models.ConvTran.hyper_parameters  as conTran_param
from models.ConvTran.Training import SupervisedTrainer, train_runner





def main():


    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset_name in ['CMJ','MP','synth_2lines','synth_1line']:
        train_X, train_y, test_X, test_y, seq_len, n_channels, n_classes = load_data(dataset_name)

        transform_data_dResnet = lambda : transform_data4ResNet(X_train= train_X, y_train= train_y,
                                             X_test= test_X, y_test= test_y,device=device, batch_s=(32,32))
        transform_data_resnet = lambda : transform2tensors(train_X,train_y,test_X,test_y, batch_size=(64,64),
                                                           device=device )


        for n in range(3):

            for norm in [True,False]:
                # con trans
                device, seed = Initialization(conTran_param.params)
                train_loader,test_loader, conTran_params = transform4ConvTran( train_X, train_y,test_X, test_y,
                                        conTran_param.params,n_classes)
                model = ConvTran(conTran_params, num_classes=n_classes).to(device)
                conTran_params['Norm'] = norm
                optimizer = torch.optim.Adam(model.parameters())

                trainer = SupervisedTrainer(model, train_loader, device,conTran_params['loss_module'],optimizer,
                                            l2_reg=0, print_interval=10, console=False, print_conf_mat=False)
                test_evaluator = SupervisedTrainer(model, test_loader, device, conTran_params['loss_module']
                                                ,optimizer,print_interval=10, console=False,print_conf_mat=False)

                file_name = "_".join( ("ConvTrans",str(n),"norm",str(norm),"seed",str(seed),".pt") )
                file_path= "/".join( ("saved_models",dataset_name,file_name) )
                train_runner(conTran_params, model, trainer, test_evaluator,optimizer=optimizer,path=file_path)
        exit()

        """
            for model_n in ["dResNet","ResNet"]:
                for n_filters in [64]:

                    if model_n=="ResNet":
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
            """



if __name__ == "__main__" :
    main()

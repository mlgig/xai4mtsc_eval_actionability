from models.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN, InceptionModel, dInceptionModel
from pytorch_utils import transform_data4ResNet, transform2tensors, transform4ConvTran
from models.MyModels.MyMiniRocket import MyMiniRocket
from load_data import load_data
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from models.ConvTran.Models.utils import load_model as load_transformer

from copy import deepcopy


from models.ConvTran.utils import Setup, Initialization
from models.ConvTran.hyper_parameters import params as transform_params
from models.ConvTran.Models.model import ConvTran
from models.ConvTran.Training import SupervisedTrainer, train_runner
from copy import deepcopy


def main():


    device = "cuda" if torch.cuda.is_available() else "cpu"


    for dataset_name in ['CMJ','MP']: #'MP','synth_2lines']:
        train_X, train_y, test_X, test_y, seq_len, n_channels, n_classes = load_data(dataset_name)

        transform_data_dResnet = lambda : transform_data4ResNet(X_train= train_X, y_train= train_y,
                                             X_test= test_X, y_test= test_y,device=device, batch_s=(32,32))
        transform_data_resnet = lambda : transform2tensors(train_X,train_y,test_X,test_y, batch_size=(64,64),
                                                           device=device )

        """
        for n in range(5):

            for model_n in ["Inception","dInception"]:
                for n_filters in [32,64]:

                    if model_n=="Inception":
                        train_loader, test_loader, enc= transform_data_resnet()
                        #resNet = ResNetBaseline(n_channels, mid_channels=n_filters,num_pred_classes= n_classes).to(device)
                        modelarch = InceptionModel(num_blocks=3, in_channels=n_channels, out_channels=n_filters,
                                               bottleneck_channels=n_filters, kernel_sizes=[10,20,40],
                                               use_residuals=True, num_pred_classes=n_classes).to(device)

                    else:
                        train_loader, test_loader, enc= transform_data_dResnet()
                        #resNet = dResNetBaseline(n_channels, mid_channels=n_filters,num_pred_classes= n_classes).to(device)
                        modelarch = dInceptionModel(num_blocks=3, in_channels=n_channels, out_channels=n_filters,
                                                       bottleneck_channels=n_filters, kernel_sizes=[10,20,40],
                                                        use_residuals=True, num_pred_classes=n_classes).to(device)



                    model = ModelCNN(model=modelarch, n_epochs_stop=50,device=device, save_path="saved_models/tmp.pt")
                    acc = model.train(num_epochs=11,train_loader=train_loader,test_loader=test_loader)

                    model_name = "_".join( (model_n, str(n_filters), str(n), str(acc)[2:5] ) )
                    file_path= "/".join( ("saved_models",dataset_name,model_name) )
                    os.rename("saved_models/tmp.pt",file_path+".pt") #torch.save(resNet, file_path+".pt")
                    torch.cuda.empty_cache()
                    print(model_n," accuracy was ",acc)
         """


        for n in range(2):
            config = deepcopy( transform_params )
            device = Initialization(config)

            train_loader, test_loader, enc = transform4ConvTran(config, n_classes, test_X, test_y, train_X, train_y)

            model = ConvTran(config, num_classes=config['num_labels']).to(device)

            config['optimizer'] = torch.optim.Adam(model.parameters())



            trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
                                        print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
            val_evaluator = SupervisedTrainer(model, test_loader, device, config['loss_module'],
                                              print_interval=config['print_interval'], console=config['console'],
                                              print_conf_mat=False)

            file_path =  "./saved_models/"+dataset_name+"/ConvTran_"+str(n)+".pt"
            train_runner(config, model, trainer, val_evaluator,file_path)



            model2 = load_transformer(model,file_path)
            model2.eval()
            output = model2( torch.tensor( test_loader.dataset.feature.astype(np.float32) ).type(torch.float32).to("cuda") )
            preds = torch.argmax(output, dim=-1)
            from sklearn.metrics import accuracy_score
            print ("ACCURACY 1",accuracy_score(test_loader.dataset.labels, preds.to("cpu").numpy() ) )


            tot_preds = []
            model2.eval()
            for batch in test_loader:
                X,y , id  = batch
                output = model2(X.to(device))
                preds = torch.argmax( output, dim = - 1).cpu().numpy()
                tot_preds.append(preds)
            tot_preds = np.concatenate(tot_preds)
            print ("ACCURACY 2",accuracy_score(test_loader.dataset.labels,tot_preds) )




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

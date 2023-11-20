from models.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN, InceptionModel, dInceptionModel
from pytorch_utils import transform_data4ResNet, transform2tensors
from models.MyModels.MyMiniRocket import MyMiniRocket
from load_data import load_data
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from models.ConvTran.Models.utils import load_model as load_transformer


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



def main():


    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset_name in ['CMJ']: #'MP','synth_2lines']:
        train_X, train_y, test_X, test_y, seq_len, n_channels, n_classes = load_data(dataset_name)
        transform_data_dResnet = lambda : transform_data4ResNet(X_train= train_X, y_train= train_y,
                                             X_test= test_X, y_test= test_y,device=device, batch_s=(32,32))
        transform_data_resnet = lambda : transform2tensors(train_X,train_y,test_X,test_y, batch_size=(64,64), device=device )


        for n in range(1):
            """
            # Inception / dInception




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











            #train_loader, test_loader, enc= transform_data_resnet()
            from models.ConvTran.utils import Setup
            args = {'data_path': 'Dataset/UEA/', 'output_dir': 'Results', 'Norm': False, 'val_ratio': 0.2,
                    'print_interval': 10, 'Net_Type': ['C-T'], 'emb_size': 16, 'dim_ff': 256, 'num_heads': 8,
                    'Fix_pos_encode': 'tAPE', 'Rel_pos_encode': 'eRPE', 'epochs': 10, 'batch_size': 16,
                    'lr': 0.001, 'dropout': 0.01, 'val_interval': 2, 'key_metric': 'accuracy', 'gpu': 0,
                    'console': False, 'seed': 1234}
            config = Setup(args)


            from utils import one_hot_encoding
            train_y,test_y,enc = one_hot_encoding(train_y,test_y)

            train_dataset = dataset_class(train_X,train_y)
            test_dataset = dataset_class(test_X,test_y)
            train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)


            from models.ConvTran.Models.model import model_factory, count_parameters

            config['num_labels'] = n_classes
            config['Data_shape'] = train_loader.dataset.feature.shape
            config['loss_module'] = torch.nn.CrossEntropyLoss(reduction='none')

            model = model_factory(config)
            model.to(device)
            config['optimizer'] = torch.optim.Adam(model.parameters())


            from models.ConvTran.Training import SupervisedTrainer, train_runner
            #'logger.info('Starting training...')
            trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
                                        print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
            val_evaluator = SupervisedTrainer(model, test_loader, device, config['loss_module'],
                                              print_interval=config['print_interval'], console=config['console'],
                                              print_conf_mat=False)

            train_runner(config, model, trainer, val_evaluator, "./tmp/"+dataset_name+".pt")

            model2 = load_transformer(model,"./tmp/"+dataset_name+".pt")
            output = model2( torch.tensor( test_loader.dataset.feature ).type(torch.float32).to("cuda") )
            preds = torch.argmax(output, dim=-1)
            from sklearn.metrics import accuracy_score
            print ("ACCURACYY",accuracy_score(test_loader.dataset.labels, preds.to("cpu").numpy() ) )
            #best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
            #best_model.to(device)

            #best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, config['loss_module'],
            #                                        print_interval=config['print_interval'], console=config['console'],
            #                                        print_conf_mat=True)
            #best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
            #print_str = 'Best Model Test Summary: '
            #for k, v in best_aggr_metrics_test.items():
            #    print_str += '{}: {} | '.format(k, v)
            #print(print_str)




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

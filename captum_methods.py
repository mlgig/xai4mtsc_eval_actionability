import os
import numpy as np
from tqdm import trange
import torch
from load_data import load_data
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    Saliency,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation
)
from pytorch_utils import transform_data4ResNet, transform2tensors, load_ConvTran
import timeit
from torch import nn
from sklearn.metrics import accuracy_score
from captum._utils.models.linear_model import *
from models.dCAM.src.explanation.DCAM import DCAM

captum_methods = {
    "gradient" : [
        {"method" :DeepLift, "require_baseline":False, "batch_size":16},
        {"method" :DeepLiftShap, "require_baseline":True, "batch_size":4},
        {"method" :IntegratedGradients, "require_baseline":False, "batch_size":4},
        {"method" :GradientShap, "require_baseline":True, "batch_size":16},
        {"method" :Saliency, "require_baseline":False, "batch_size":16}
    ],
    "permutation": [
        {"method" : FeatureAblation, "require_baseline":False, "batch_size":32},
        {"method" : FeaturePermutation, "require_baseline":False, "batch_size":32},
        {"method" :KernelShap, "require_baseline":False, "batch_size":1},
        {"method" :ShapleyValueSampling, "require_baseline":False, "batch_size":32}
    ]
}


def load_dataset(device, groups, model_name, test_X, test_y, train_X, train_y):

    if model_name.startswith("dResNet"):
        groups, _,_,_,_= transform_data4ResNet(X_train=groups,y_train=[1], X_test=np.array([[1,2]]) ,y_test=[1],
                                               device=device, batch_s=None)
        _, _, X_test, y_test, enc = transform_data4ResNet( X_train=train_X , X_test=test_X, y_test=test_y,
                                                           y_train=train_y, device=device,batch_s=None)

    else:
        X_train, y_train, X_test, y_test, enc = transform2tensors(X_train=train_X, y_train=train_y,
                                                                  X_test=test_X, y_test=test_y, device=device)
        groups = torch.tensor(groups, device=device)

    return X_test, y_test, groups.type(torch.int64).to(device), enc


def get_groups(n_chunks,n_channels,series_length):
    groups = np.array([[i + j * n_chunks for i in range(n_chunks)] for j in range(n_channels)])
    groups = np.expand_dims(np.repeat(groups, np.ceil(series_length / n_chunks).astype(int), axis=1), 0)[:, :, :series_length]
    return groups

def compute_outputs( model, X_test, y_test, batch_size, to_save,device):

    # get time series length and temp data structures
    n_instances = X_test.shape[0]
    predictions = []
    ground_truths = []

    for i in range(0, n_instances, batch_size):
        # computing outputs and append along gt labels
        samples = (X_test[i:(min(i + batch_size, n_instances))]).to(device)
        labels = y_test[i:(min(i + batch_size, n_instances))].to(device)
        output = model(samples)
        predictions.append(torch.argmax(output, dim=1).cpu().numpy())
        ground_truths.append(labels.cpu().numpy())

    #concatenate and save
    predictions = np.concatenate(predictions)
    ground_truths = np.concatenate(ground_truths)
    to_save["predicted_labels"] = predictions
    to_save["ground_truth_labels"] = ground_truths
    return accuracy_score(ground_truths,predictions)



def running_captum_XAI(X_test,y_test,  model, model_name, device,  attributions,chunking, groups,method_dict) :

    # get current method info
    method_name = str(method_dict["method"]).split(".")[-1][:-2]
    explainer = method_dict["method"](model)

    #batch_size = method_dict["batch_size"]*4 if method_name not in ["KernelShap","Lime"] else method_dict["batch_size"]
    batch_size =  method_dict["batch_size"]

    explanations = []
    n_instances = X_test.shape[0]

    start = timeit.default_timer()
    for i in trange(0, n_instances, batch_size):
        # take samples, label and baseline (same shape of the samples)
        samples = (X_test[i:(min(i + batch_size, n_instances))]).clone().detach().requires_grad_(True).to(device)
        labels = y_test[i:(min(i + batch_size, n_instances))].clone().detach().to(device)
        baseline = torch.normal(mean=0, std=1, size=samples.shape).to(device)

        # set kwargs
        kwargs = {}
        if method_dict["require_baseline"]:
            kwargs["baselines"] = baseline
        if method_dict in captum_methods["permutation"] and chunking:
            kwargs["feature_mask"] = groups
        if method_name=="Saliency":
            kwargs["abs"] = False

        # get the explanation and save it
        explanation = explainer.attribute(samples, target=labels, **kwargs)
        explanations.append(explanation.cpu().detach().numpy()[:, 0]) if model_name.startswith("dResNet") \
            else explanations.append(explanation.cpu().detach().numpy())

    # measure time took by the method
    end = timeit.default_timer()
    attributions[method_name] = np.concatenate(explanations)
    print("\t", model_name, method_name, "took", (end - start), " seconds")


def run_dCAM(model,device,attributions, dataset_name,X_test, y_test):
    last_conv_layer = model._modules['0'].layers[2]
    fc_layer_name = model._modules['0'].final
    dCAM = DCAM(model,device,last_conv_layer,fc_layer_name)

    nb_permutation = 6 if dataset_name=="CMJ" else 200
    generate_all = True if dataset_name=="CMJ" else False

    outupts = []
    nb_err = 0

    start = timeit.default_timer()
    for i in trange(X_test.shape[0]):
        instance = X_test[i][0].cpu().numpy()
        label = y_test[i]
        try:
            dcam,permutation_success = dCAM.run(
                instance=instance, nb_permutation=nb_permutation, label_instance=label,generate_all=generate_all)
            outupts.append(dcam)
        except IndexError:
            outupts.append( np.zeros(shape=instance.shape))
            nb_err+=1

    end = timeit.default_timer()
    attributions['dCAM'] = np.stack(outupts)
    print("dCAM took", (end-start), "seconds and couldn't find an explanation for ",nb_err,"instance")

def main():


    device ="cuda" if torch.cuda.is_available() else "cpu"

    # TODO for loop for every dataset?
    for dataset_name in ['MP50']:
        train_X,train_y,test_X,test_y, seq_len, n_channels, n_classes = load_data(dataset_name)

        for n_chunks in [-1]:

            for model_name in ['dResNet']:

                # load the current model and transform the data accordingly
                groups = get_groups(n_chunks,n_channels,seq_len)

                saved_model_name = os.path.join( "saved_models" ,dataset_name ,model_name+".pt")
                if model_name.lower().count('convtran')>0:
                    model, test_loader, enc ,device = load_ConvTran(test_X=test_X,test_y=test_y, train_X=None,
                        train_y=train_y,n_classes=n_classes,path=saved_model_name)

                    groups = torch.tensor(groups, device=device)
                    X_test =torch.tensor(test_loader.dataset.feature).type(torch.float32)
                    y_test = torch.tensor(test_loader.dataset.labels).type(torch.int64)
                else:
                    model =  torch.load( saved_model_name )
                    X_test, y_test, groups, enc = load_dataset(device, groups, model_name, test_X, test_y, train_X,train_y)

                model = nn.Sequential(model, nn.Softmax(dim=-1))


                methods2use = captum_methods["gradient"] + captum_methods["permutation"] if \
                    (model_name in ["ConvTran","ResNet"] and n_chunks==-1) else  captum_methods["permutation"]

                # dict for the attributions
                attributions = {}
                to_save = {"explanations": attributions, "classes_idx" : enc.classes_,
                           "ground_truth_labels":[],"predicted_labels":[]}
                accuracy = compute_outputs( model, X_test, y_test,batch_size=8, to_save=to_save, device=device)
                print("\n",model_name,n_chunks, "accuracy is",accuracy)

                if model_name=='dResNet' and n_chunks==-1:
                    run_dCAM(model,device,attributions,dataset_name, X_test, y_test)
                for method_dict in methods2use:
                    running_captum_XAI(X_test,y_test, model, model_name, device,attributions,n_chunks!=-1, groups,method_dict)

                path = os.path.join( "./explanations",dataset_name )
                file_name = "_".join( (model_name,"n_chunks",str(n_chunks),"saliency" )  )+".npy"
                np.save( os.path.join(path,file_name),to_save)




if __name__ == "__main__" :
    main()

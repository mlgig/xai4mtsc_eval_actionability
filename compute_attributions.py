import os
import numpy as np
import timeit
import torch
import argparse
from tqdm import trange
from load_data import load_data
from pytorch_utils import transform_data4ResNet, transform2tensors, load_ConvTran
from sklearn.metrics import accuracy_score
from dCAM import run_dCAM
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Saliency,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation
)

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
    """
    function to convert dataset into the appropriate format
    """
    if model_name.startswith("dResNet"):
        groups, _,_,_,_= transform_data4ResNet(X_train=groups,y_train=[1], X_test=np.array([[1,2]]) ,y_test=[1],
                                         batch_s=None)
        _, _, X_test, y_test, enc = transform_data4ResNet( X_train=train_X , X_test=test_X, y_test=test_y,
                                                           y_train=train_y, batch_s=None)

    else:
        X_train, y_train, X_test, y_test, enc = transform2tensors(X_train=train_X, y_train=train_y,
                                                                  X_test=test_X, y_test=test_y, device=device)
        groups = torch.tensor(groups, device=device)

    return X_test, y_test, groups.type(torch.int64).to(device), enc


def get_groups(n_chunks,n_channels,series_length):
    """
	function returning how to group time points into time Series accordingly to the given arguments
	To be noted that it operates channel-wise i.e. each channel is divided into "n_chunks" chunks

	:param n_chunks:        number of chunks to be used
	:param n_channels:      number of channel of each instance in the dataset
	:param series_length:   length of each channel of each instance in the dataset
	:return:                a numpy array representing how to group the time points
	"""
    groups = np.array([[i + j * n_chunks for i in range(n_chunks)] for j in range(n_channels)])
    groups = np.expand_dims(np.repeat(groups, np.ceil(series_length / n_chunks).astype(int), axis=1), 0)[:, :, :series_length]
    return groups

def compute_outputs( model, X_test, y_test, batch_size, to_save,device):
    """
    function to compute the output of the current model for each instance in the dataset
    :param model:       the classifier to be used
    :param X_test:      instances
    :param y_test:      labels for the corresponding instances
    :param batch_size:  which batch size to be used
    :param to_save:     already defined data structure
    :param device:      which device to use for the forward
    :return:            accuracy of the model
    """

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

    # concatenate and save
    predictions = np.concatenate(predictions)
    ground_truths = np.concatenate(ground_truths)
    to_save["predicted_labels"] = predictions
    to_save["ground_truth_labels"] = ground_truths

    # compute and return also the accuracy
    acc = accuracy_score(ground_truths,predictions)
    return acc



def running_captum_XAI(X_test,y_test,  model, model_name, device,  attributions,chunking, groups,method_dict) :
    """
    function to run the selected attribution methods, all implemented using captum
    :param X_test:          instances to explain
    :param y_test:          labels for the instances
    :param model:           which model to explain
    :param model_name:      name of the model to explain
    :param device:          device to be used
    :param attributions:    already defined data structure
    :param chunking:        whether to use chunking
    :param groups:          tensor telling how to group time points for chunking
    :param method_dict:     which method to used and their parameters
    :return:                updated attributions data structure containing new computed saliency maps
    """

    # get current method info
    method_name = str(method_dict["method"]).split(".")[-1][:-2]
    explainer = method_dict["method"](model)

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
        # Saliency by default return absolute value, fix it
        if method_name=="Saliency":
            kwargs["abs"] = False

        # get the explanation and append to previously defined list
        explanation = explainer.attribute(samples, target=labels, **kwargs)
        explanations.append(explanation.cpu().detach().numpy()[:, 0]) if model_name.startswith("dResNet") \
            else explanations.append(explanation.cpu().detach().numpy())

    # measure elapsed time
    end = timeit.default_timer()
    attributions[method_name] = np.concatenate(explanations)
    print("\n\t", model_name, method_name, "took", (end - start), " seconds \n\n")
    return  attributions



def main(args):

    # get device to be used and parameters
    device ="cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset
    ns_chunks = args.ns_chunks
    model_name = args.classifier

    # load data
    train_X,train_y,test_X,test_y, seq_len, n_channels, n_classes = load_data(dataset_name)

    for n_chunks in ns_chunks:

        # load the current model and groups tensor (i.e. how to concatenate time points for chunking)
        # then transform the data and groups tensor accordingly to the chosen model
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

        # on top of every model put a softmax
        model =  torch.nn.Sequential(model, torch.nn.Softmax(dim=-1))

        methods2use = captum_methods["gradient"]+ captum_methods["permutation"] if \
            (model_name in ["ConvTran","ResNet"] and n_chunks==-1) else  captum_methods["permutation"]

        # instantiate dictionary to save attributions and metadata
        attributions = {}
        to_save = {"explanations": attributions, "classes_idx" : enc.classes_, "ground_truth_labels":[],"predicted_labels":[]}
        accuracy= compute_outputs( model, X_test, y_test,batch_size=16, to_save=to_save, device=device)
        print("\n",model_name,n_chunks, "accuracy is",accuracy)

        # if model is dResNet operating point-wise run also dCAM
        if model_name=='dResNet' and n_chunks==-1:
            run_dCAM(model,device,attributions,dataset_name, X_test, y_test)
        for method_dict in methods2use:
            running_captum_XAI(X_test,y_test, model, model_name, device,attributions,n_chunks!=-1, groups,method_dict)

        # save previously defined dictionary into a .npy file
        path = os.path.join( "./explanations",dataset_name )
        file_name = "_".join( (model_name,"n_chunks",str(n_chunks))  )+".npy"
        np.save( os.path.join(path,file_name),to_save)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="which dataset to be used")
    parser.add_argument("classifier", type=str, help="which classifier to be trained")
    parser.add_argument("ns_chunks", type=int, nargs='+', help="which number of chunks to be used (-1 equal"
                                                              " to not chunking i.e. explain point-wise")
    args = parser.parse_args()
    main(args)


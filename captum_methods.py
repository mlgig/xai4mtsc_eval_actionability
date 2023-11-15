import os
import numpy as np
from tqdm import trange
import torch
from load_data import load_data
from captum.attr import (
    LRP,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    NoiseTunnel,
    Saliency,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation
)
from pytorch_utils import transform_data4ResNet, transform2tensors
import timeit
from torch import nn
from captum._utils.models.linear_model import *

captum_methods = {
    "gradient" : [
        {"method" :DeepLift, "require_baseline":False, "batch_size":16},
        {"method" :DeepLiftShap, "require_baseline":True, "batch_size":8},
        {"method" :IntegratedGradients, "require_baseline":False, "batch_size":4},
        {"method" :GradientShap, "require_baseline":True, "batch_size":16},
        {"method" :Saliency, "require_baseline":False, "batch_size":16}
    ],
    "permutation": [
        {"method" :ShapleyValueSampling, "require_baseline":False, "batch_size":32},
        {"method" : FeatureAblation, "require_baseline":False, "batch_size":32},
        {"method" : FeaturePermutation, "require_baseline":False, "batch_size":32},
        {"method" :KernelShap, "require_baseline":False, "batch_size":1},
        {"method" :Lime, "require_baseline":False, "batch_size":1},
    ]
}

limit = 1000
selected = np.load("explanations/synth_2lines/selected_idxs.npy")[:1000]

# TODO do I need it??
def get_grouping4dResNet(sample):
    groups = torch.tensor( [i for i in range( sample.size)])
    return groups.resize(1,8,500)


def main():

    # TODO take only one row in dResNet explanations
    # set device, load data
    device ="cuda" if torch.cuda.is_available() else "cpu"
    # TODO for loop for every dataset?
    dataset_name="synth_2lines"
    train_X,train_y,test_X,test_y, seq_len, n_channels, n_classes = load_data(dataset_name)
    test_X = test_X[selected] ; test_y = test_y[selected]

    for n_chunks in [5,10,15,20]:
        for model_name in [ "dResNet.pt", "ResNet.pt","MiniRocket.pt"] :
            # load the current model and transform the data accordingly
            model = torch.load( os.path.join( "saved_models" ,dataset_name ,model_name) )
            model = nn.Sequential(model, nn.Softmax(dim=-1))

            # TODO everything is hardcoded here: n_chunks?, 384,3 etc.
            groups = np.array( [ [i+j*n_chunks for i in range(n_chunks)] for j in range(8)] )
            groups = np.expand_dims( np.repeat(groups,np.ceil( 500 /n_chunks).astype(int),axis=1), 0)[:,:,:500]

            if model_name.startswith("dResNet"):
                dResNet_groups = transform_data4ResNet(groups,y_test=None, device=device,batch_s=None)
                _,_,X_test,y_test, enc = transform_data4ResNet( X_test= test_X,y_test=test_y,y_train=train_y,device=device, batch_s=None)
                #groups = get_grouping4dResNet(sample)
                #groups = transform_data4ResNet(groups,y_test=None,device=device,batch_s=None)
            else:
                X_train,y_train,X_test,y_test, enc = transform2tensors(X_train=train_X,y_train= train_y,
                                                                       X_test=test_X,y_test=test_y,device=device)
                groups = torch.tensor( groups,device=device)

            # dict for the attributions
            attributions = {}
            ground_truths= []
            predictions = []
            to_save = {"explanations": attributions, "ground_truth_labels":None,"predicted_labels":None}
            methods2use = captum_methods["gradient"]+ captum_methods["permutation"] if model_name.startswith("ResNet") else (
                captum_methods["permutation"] )

            for method_dict in methods2use:

                # TODO to move
                # init misc
                method_name = str(method_dict["method"]).split(".")[-1][:-2]
                explainer = method_dict["method"](model)
                batch_size = method_dict["batch_size"]
                explanations = []

                start = timeit.default_timer()
                for i in trange(0,limit,batch_size):

                    # take samples, label and baseline (same shape of the samples)
                    samples = ( X_test[i:(min(i+batch_size,limit) )] ).clone().detach().requires_grad_(True).to(device)
                    labels =  y_test[i:(min(i+batch_size,limit) )].clone().detach().to(device)
                    baseline = torch.normal(mean=0, std=1, size=samples.shape).to(device)

                    # save ground truth label and model prediction
                    # TODO to do only at first time
                    if (to_save["ground_truth_labels"] is None and to_save["predicted_labels"] is None):
                        output = model(samples)
                        predictions.append(torch.argmax(output,dim=1).cpu().numpy())
                        ground_truths.append(labels.cpu().numpy())

                    # set kwargs
                    kwargs = {}
                    if method_dict["require_baseline"]:
                        kwargs["baselines"] = baseline
                    if method_dict in captum_methods["permutation"]:
                        if  model_name.startswith("dResNet"):
                            kwargs["feature_mask"] = dResNet_groups
                        else:
                            kwargs["feature_mask"] =groups

                    # get the explanation and save it
                    explanation = explainer.attribute(samples, target=labels, **kwargs)
                    explanations.append( explanation.cpu().detach().numpy()[:,0] ) if model_name.startswith("dResNet") \
                        else explanations.append( explanation.cpu().detach().numpy() )

                # measure time took by the method
                end = timeit.default_timer()
                to_save['ground_truth_labels'] = np.concatenate(ground_truths) ; to_save['predicted_labels']= np.concatenate(predictions)
                attributions[method_name] = np.concatenate(explanations)
                print( "explaining the instance ", i, " using ", method_dict["method"] , "took",(end-start), " seconds\n",
                       "in shape:",samples.shape,"out shaape:", explanations[-1].shape,"\n")

            # TODO hard coded dataset name!
            np.save("./explanations/synth_2lines/"+model_name+"_"
                    +str(limit)+"_"+str(n_chunks)+"_8chunks.npy",to_save)


if __name__ == "__main__" :
    main()

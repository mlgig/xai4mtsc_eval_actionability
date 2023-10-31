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

captum_mthods = {
    "gradient" : [
        {"method" :DeepLift, "require_baseline":False, "batch_size":16},
        {"method" :DeepLiftShap, "require_baseline":True, "batch_size":8},
        {"method" :IntegratedGradients, "require_baseline":False, "batch_size":4},
        {"method" :GradientShap, "require_baseline":True, "batch_size":16},
        {"method" :Saliency, "require_baseline":False, "batch_size":16}
    ],
    "permutation": [
        {"method" : FeatureAblation, "require_baseline":False, "batch_size":8},
        {"method" : FeaturePermutation, "require_baseline":False, "batch_size":16},
        {"method" :KernelShap, "require_baseline":False, "batch_size":1},
        {"method" :Lime, "require_baseline":False, "batch_size":1},
        {"method" :ShapleyValueSampling, "require_baseline":False, "batch_size":16}
    ]
}

limit = 300

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


    for model_name in [ "ResNet.pt" ,"dResNet.pt"]:   # ,"dResNet.pt"
        # load the current model and transform the data accordingly
        model = torch.load( os.path.join( "saved_models" ,dataset_name ,model_name) )
        model = nn.Sequential(model, nn.Softmax(dim=-1))

        # TODO not hard coded!
        n_chunks = 50
        groups = np.array( [ [i+j*n_chunks for i in range(n_chunks)] for j in range(8)] )
        groups = np.expand_dims( np.repeat(groups,10,axis=1), 0)
        print ( np.unique(groups ))

        if model_name.startswith("ResNet"):
            X_train,y_train,X_test,y_test, enc = transform2tensors(train_X,train_y,test_X,test_y,device=device)
            groups = torch.tensor( groups,device=device)

        elif model_name.startswith("dResNet"):
            dResNet_groups = transform_data4ResNet(groups,y_test=None, device=device,batch_s=None)
            _,_,X_test,y_test, enc = transform_data4ResNet(test_X,test_y,y_train=train_y,device=device, batch_s=None)
            #groups = get_grouping4dResNet(sample)
            #groups = transform_data4ResNet(groups,y_test=None,device=device,batch_s=None)

        # dict for the attributions
        attributions = {}
        # TODO delete following 2 lines
        ground_truths= []
        predictions = []
        to_save = {"explanations": attributions, "ground_truth_labels":None,"predicted_labels":None}
        methods2use = captum_mthods["permutation"]

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
                samples = ( X_test[i:(i+min(batch_size,limit) )] ).clone().detach().requires_grad_(True).to(device)
                labels =  y_test[i:(i+min(batch_size,limit) )].clone().detach().to(device)
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
                if  model_name.startswith("dResNet"):
                    kwargs["feature_mask"] = dResNet_groups
                else:
                    kwargs["feature_mask"] =groups

                # get the explanation and save it
                explanation = explainer.attribute(samples, target=labels, **kwargs)#,feature_mask=groups)#, baselines=baseline)
                explanations.append(explanation.cpu().detach().numpy())

            # measure time took by the method
            to_save['ground_truth_labels'] = np.concatenate(ground_truths) ; to_save['predicted_labels']= np.concatenate(predictions)
            end = timeit.default_timer()
            attributions[method_name] = np.concatenate(explanations)
            print( "explaining the instance ", i, " using ", method_dict["method"] , "took",(end-start), " seconds\n",
                   "in shape:",samples.shape,"out shape:", explanation.shape,"\n")

            np.save("./explanations/synth_2lines/"+model_name+"_"+str(limit)+"_"+str(n_chunks)+"_8chunks.npy",to_save)
        """
        # explainin in chunks: chunk 0 from 0 t 100
        grouping = torch.zeros(1,3,384).type(torch.int64)
        grouping[:,:,100:200] = 1   #chunk 1 from 100 t 200
        grouping[:,:,200:300] = 2   #chunk 2 from 200 t 300
        grouping[:,:,300:] = 3      #chunk 3 from 300 on

        start = timeit.default_timer()
        explanation_grouped = kernel_shap.attribute(samples, target=labels, baselines=baseline, feature_mask=grouping)
        end = timeit.default_timer()
        print( "grouped explanation: in shape:", samples.shape, "out shape:", explanation_grouped.shape,
               " but n. different elements", len(torch.unique(explanation_grouped)), "time was ", (end-start) ),"\n\n\n"
        """

if __name__ == "__main__" :
    main()


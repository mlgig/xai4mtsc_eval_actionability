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

captum_mthods = [
    {"method" :ShapleyValueSampling, "require_baseline":False, "batch_size":4},
    #{"method" :DeepLift, "require_baseline":False, "batch_size":16},
    #{"method" :DeepLiftShap, "require_baseline":True, "batch_size":16},
    #{"method" :IntegratedGradients, "require_baseline":False, "batch_size":4},
    ##{"method" :GradientShap, "require_baseline":True, "batch_size":16},
    #{"method" :Saliency, "require_baseline":False, "batch_size":16},
    #{"method" :KernelShap, "require_baseline":False, "batch_size":1},
    #{"method" :Lime, "require_baseline":False, "batch_size":1},
    #{"method" : FeatureAblation, "require_baseline":False, "batch_size":8},
    #{"method" : FeaturePermutation, "require_baseline":False, "batch_size":16},
]

def get_grouping4dResNet(sample):
    groups = torch.tensor( [i for i in range( sample.size)])
    # TODO get why it's not working
    return groups.resize(1,8,500)

def main():

    # set device, load and transform data
    device ="cuda" if torch.cuda.is_available() else "cpu"

    # TODO for loop for every dataset?
    dataset_name="synth_2lines"
    X_train,y_train,X_test,y_test, seq_len, n_channels, n_classes = load_data(dataset_name)

    for model_name in [ "ResNet.pt", ]:     # "dResNet_64_0_910.pt"
        model = torch.load( os.path.join( "saved_models" ,dataset_name ,model_name) )

        if model_name.startswith("ResNet"):
            X_train,y_train,X_test,y_test, enc = transform2tensors(X_train,y_train,X_test,y_test,device=device)
            # loading the model
        elif model_name.startswith("dResNet"):
            sample = X_train[0]
            X_train,y_train,X_test,y_test, enc = transform_data4ResNet(X_train,y_train,X_test,y_test,device=device, batch_s=None)
            groups = get_grouping4dResNet(sample)
            # TODO only test mode!
            groups = transform_data4ResNet(groups,
                    torch.tensor([0]),groups,torch.tensor( [0]),device=device, batch_s=None)[0].type(torch.int64).to(device)


        baseline = torch.normal(mean=0, std=1, size=X_test[:2].shape).to(device)
        #expected_value = model(baseline.to(device))

        attributions = {}
        # TODO do not forget about ShapleyValueSampling !
        for method_dict  in captum_mthods:
            method_name = str(method_dict["method"]).split(".")[-1][:-2]

            explainer = method_dict["method"](model)
            batch_size = method_dict["batch_size"]
            explanations = []

            start = timeit.default_timer()

            for i in trange(0,100,batch_size):
                # TODO exactly 100 samples!
                samples = ( X_test[i:(i+batch_size )] ).clone().detach().requires_grad_(True).to(device)
                labels =  y_test[i:(i+batch_size )].clone().detach().to(device) # labels has to be torch.int64 (also called torch.long)

                # TODO use kwargs!
                # explaining all points in all channels

                if method_dict["require_baseline"]:
                    if  model_name.startswith("dResNet"):
                        explanation = explainer.attribute(samples, target=labels, baselines=baseline,feature_mask=groups)
                    else:
                        explanation = explainer.attribute(samples, target=labels, baselines=baseline)

                else:
                    if  model_name.startswith("dResNet"):
                        explanation = explainer.attribute(samples, target=labels,feature_mask=groups)
                    else:
                        groups = torch.tensor(  [ [i+j for i in range(10) ] for j in range(8)]).type(torch.long)
                        groups = torch.repeat_interleave(groups,50,dim=1).to(device)
                        explanation = explainer.attribute(samples, target=labels,feature_mask=groups)#, baselines=baseline)
                # in any case
                explanations.append(explanation.cpu().detach().numpy())

            end = timeit.default_timer()
            attributions[method_name] = np.concatenate(explanations)
            print( "explaining the instance ", i, " using ", method_dict["method"] , "took",(end-start), " seconds\n",
                   "in shape:",samples.shape,"out shape:", explanation.shape,"\n")

        # TODO save in a separate folder
        np.save("explanations100.npy",attributions)
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


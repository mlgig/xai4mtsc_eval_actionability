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
    FeaturePermutation
)
from pytorch_utils import transform_data4ResNet, transform2tensors
import timeit


def main():

    # set device, load and transform data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train,y_train,X_test,y_test, seq_len, n_channels, n_classes = load_data("CMJ")
    X_train,y_train,X_test,y_test, enc = transform2tensors(X_train,y_train,X_test,y_test,device=device)

    # loading the models
    for saved_model_name in ["MiniRocket_0_949",  "ResNet_128_3_949",  "Rocket_4_921"]:
        model = torch.load("saved_models/CMJ/"+saved_model_name+".pt",
            map_location=device) # using the map_location argument is possible to freely load your model in CPU or GPU
            # regardless of the device used to train it
        baseline = torch.normal(mean=0, std=1, size=X_test[:1].shape).to(device)
        expected_value = model(baseline.to(device))
        kernel_shap = KernelShap(model)

        for i in range(3):
            samples = X_test[i:(i+1)]
            labels =  y_test[i:(i+1)]       # labels has to be torch.int64 (also called torch.long)

            start = timeit.default_timer()

            # explaining all points in all channels
            explanation = kernel_shap.attribute(samples, target=labels, baselines=baseline)
            end = timeit.default_timer()
            print( "explaining the instance ", i, "-th,using model ",saved_model_name, "took",(end-start), " seconds\n",
                   "in shape:",samples.shape,"out shape:", explanation.shape,"\n")


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

if __name__ == "__main__" :
    main()


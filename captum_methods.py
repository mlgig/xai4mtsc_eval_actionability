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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train,y_train,X_test,y_test, seq_len, n_channels, n_classes = load_data("CMJ")
    X_train,y_train,X_test,y_test, enc = transform2tensors(X_train,y_train,X_test,y_test,device=device)
    #X_train,y_train,X_test,y_test, enc = transform_data4ResNet(X_train,y_train,X_test,y_test,device=device)

    for saved_model_name in ["MiniRocket",  "ResNet",  "Rocket"]:

        model = torch.load("saved_models/CMJ/"+saved_model_name+".pt", map_location=device)
        baseline = torch.normal(mean=0, std=1, size=X_test[:1].shape).to(device)
        expected_value = model(baseline.to(device))

        for i in range(10):
            samples = X_test[i:(i+1)]
            labels =  y_test[i:(i+1)]       #labels has to be torch.int64

            start = timeit.default_timer()
            kernel_shap = KernelShap(model)

            grouping = torch.zeros((1,3,384)).type(torch.int64).to(device)
            grouping[0,1,:] = 1
            grouping[0,2,:] = 2


            explanation = kernel_shap.attribute(samples, target=labels, baselines=baseline)
            end = timeit.default_timer()
            print( "explaining the instance ", i, "-th,using model ",saved_model_name, "took"
                    ,(end-start), " seconds\n",samples.shape, explanation.shape)
            print("output", len(torch.unique(explanation)) , explanation.shape )


if __name__ == "__main__" :
    main()


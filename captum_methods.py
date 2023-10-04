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
from utils import transform_data4ResNet
import timeit

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _,_,X_test,y_test, seq_len, n_channels, n_classes = load_data("tmp_synth_1line")


    miniRocket = torch.load("saved_models/MiniRocket_1line_CS10.pt")
    baseline = torch.normal(mean=0, std=1, size=X_test[:2].shape).to("cuda")
    expected_value = miniRocket(baseline.to("cuda"))
    print(expected_value)

    samples = torch.tensor(X_test[100:102] ).to("cuda")
    labels =  torch.Tensor( y_test[100:102].astype(int) ).type(torch.int64).to(device)

    #out = miniRocket(samples)
    #torch.autograd.grad(torch.unbind(out), samples)

    #kwargs = {'baselines':baseline, 'return_convergence_delta':True}
    #i = IntegratedGradients(miniRocket,multiply_by_inputs=True)
    #out = i.attribute(samples, target=labels)#, baselines=baseline, return_convergence_delta=True)
    print("\ncaptum:")
    start = timeit.default_timer()
    deeplift = ShapleyValueSampling(miniRocket)#,multiply_by_inputs = True)
    tmp = deeplift.attribute(samples, target=labels)#, baselines=baseline)
    #shap = ShapleyValueSampling(miniRocket)#, perturbations_per_eval=4)
    #res = shap.attribute(samples,target=labels)
    #metohd = LRP(miniRocket)
    #res = metohd.attribute(samples, target=labels)
    print(tmp.shape, (timeit.default_timer() - start))
    """
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test.astype(int)).to(device)
    resNet = torch.load("saved_models/resNet.pt")
    samples = X_test[:2]
    labels = y_test[:2]
    out = resNet.model( samples )

    i =  ShapleyValueSampling(resNet.model) #DeepLift(resNet.model,multiply_by_inputs=True) #IntegratedGradients(resNet.model,multiply_by_inputs=True)
    attr = i.attribute(samples, target=labels)
    print( X_test[:2].shape,"\n\noutput:", out.shape ,out, "\n\nattribute:",attr.shape)
    """

if __name__ == "__main__" :
    main()


"""
    "integrated_gradients": {
        "captum_method": IntegratedGradients,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        "noback_cudnn": True,
        "batch_size": 8,
    },
"""
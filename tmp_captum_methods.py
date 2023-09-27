from load_data import load_data
from utils import transform_data4ResNet
import torch
from sklearn.metrics import accuracy_score
from method_arguments import dict_method_arguments

# TODO better import
# importing srcs for dResNet and dCAM
import sys
base_path="./dCAM/src/"
sys.path.insert(0, base_path+'explanation')
sys.path.insert(0, base_path+'models')
from CNN_models import *
from DCAM import DCAM


# TODO check how resNet is working (final layer)

#TODO extract a fucntion
data = load_data("tmp_synth_2lines")
train_dataloader,test_dataloader, n_channels, n_classes, device,enc = transform_data4ResNet(data)
path = "../../Trang/first_experiment/saved_model/resNet/"
file="my_synth_concat_False"

# TODO back to GPU
modelarch = torch.load(path+file, map_location="cpu")
#TODO back to device=cuda
resnet = ModelCNN(model=modelarch ,n_epochs_stop=30,device="cpu")#device)#,save_path='saved_model/resNet/'#+dataset_name+"_nFilters_"+str(mid_channels)+"_"+str(i))
#cnn_output = resnet.predict( test_dataloader )

# convert back to symbolic representation and get accu racy
#symbolic_output = enc.inverse_transform(cnn_output)
#print("accuracy is",accuracy_score( symbolic_output,data['test']['y']) )


labels = torch.Tensor( test_dataloader.dataset.labels[:2]).type(torch.int64)#.to("cuda")
samples = torch.tensor( test_dataloader.dataset.samples[:2], requires_grad=True).type(torch.float) #.to("cuda")
baseline = torch.normal(mean=0, std=1, size=samples[:2].shape, requires_grad=True)#.to("cuda")

for i,method in enumerate(dict_method_arguments.keys()):
    arguments = dict_method_arguments[method]
    if i>=4:
        explainer = arguments['captum_method'](resnet.model)
    else:
        explainer = arguments['captum_method'](resnet.model, **arguments['kwargs_method'])
    # TODO re-add to cuda


    #expected_value = (
    #    resnet.model(baseline.type(torch.float32).to(device))
    #    .detach()
    #    .cpu()
    #    .numpy()
    #)   #torch.nn.functional.softmax(output)

    kwargs = {'baselines':baseline, 'return_convergence_delta':True}
    if i==2 or i==3:
        aa = explainer.attribute(samples, target=labels,**kwargs )
        print(method, samples.shape,aa[0].shape,aa[1].shape)
    else:
        aa = explainer.attribute(samples, target=labels)#,**kwargs )
        print(method, samples.shape,aa.shape)




    """
            elif type_baseline == "random":
            # return baseline as random values
            baseline = torch.normal(mean=0, std=1, size=s[:1].shape)
            expected_value = (
                self.model(baseline.type(torch.float32).to(device))
                .detach()
                .cpu()
                .numpy()
            )
    """
